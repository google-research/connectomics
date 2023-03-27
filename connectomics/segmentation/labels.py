# coding=utf-8
# Copyright 2022-2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Routines for manipulating numpy arrays of segmentation data."""

import collections
from typing import Iterable, Optional, Sequence

import edt
import numpy as np
import skimage.morphology
import skimage.segmentation


def relabel(labels: np.ndarray, orig_ids: Iterable[int],
            new_ids: Iterable[int]) -> np.ndarray:
  """Relabels `labels` by mapping `orig_ids` to `new_ids`.

  Args:
    labels: ndarray of segment IDs
    orig_ids: iterable of existing segment IDs
    new_ids: iterable of new segment IDs (len(new_ids) == len(orig_ids))

  Returns:
    int64 ndarray with updated segment IDs
  """
  orig_ids = np.asarray(orig_ids)
  new_ids = np.asarray(new_ids)
  assert orig_ids.size == new_ids.size

  relabel_hashtable = {
      new_id: orig_id for new_id, orig_id in zip(orig_ids, new_ids)
  }
  relabeled = [relabel_hashtable[l] for l in labels.flatten()]
  return np.asarray(relabeled).reshape(labels.shape)


def make_contiguous(
    labels: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
  """Relabels 'labels' so that its ID space is dense.

  If N is the number of unique ids in 'labels', the new IDs will cover the range
  [0..N-1].

  Args:
    labels: ndarray of segment IDs

  Returns:
    tuple of:
      ndarray of dense segment IDs
      list of (old_id, new_id) pairs
  """
  orig_ids = np.unique(np.append(labels, np.uint64(0)))
  new_ids = np.arange(len(orig_ids))
  return relabel(labels, orig_ids, new_ids), list(zip(orig_ids, new_ids))


def are_equivalent(label_a: np.ndarray, label_b: np.ndarray) -> bool:
  """Returns whether two volumes contain equivalent segmentations.

  Segmentations are considered equivalent when there exists a 1:1 map between
  the two arrays -- that is segment shapes and locations have to be exactly
  the same but their IDs can be shuffled.

  Args:
    label_a: numpy array of segments
    label_b: numpy array of segments

  Returns:
    True iff the segmentations 'label_a' and 'label_b' are equivalent.
  """
  if label_a.shape != label_b.shape:
    return False

  a_to_b = {}
  b_to_a = {}

  for a, b in set(zip(label_a.flat, label_b.flat)):
    if a not in a_to_b:
      a_to_b[a] = b
    if b not in b_to_a:
      b_to_a[b] = a

    if a_to_b[a] != b:
      return False

    if b_to_a[b] != a:
      return False

  return True


def erode(labels: np.ndarray, radius: int = 2, min_size: int = 50):
  """Creates empty spaces between segments, while preserving small segments.

  Args:
    labels: 2D or 3D ndarray of uint64
    radius: radius to use for erosion operation; 2*r + 1 pixels will separate
      neighboring segments after erosion, unless the segments are small, in
      which case the separation might be smaller
    min_size: erosion will not be applied to segments smaller than this value

  Returns:
    ndarray with eroded segments; shape is the same as 'labels'
  """
  # Nothing to do.
  if np.all(labels == 0):
    return labels.copy()

  assert len(labels.shape) in (2, 3)
  sizes = collections.Counter(labels.flat)
  small_segments = [
      segment_id for segment_id, size in sizes.items() if size <= min_size
  ]
  eroded = labels.copy()
  is_2d = len(labels.shape) == 2

  # Introduce a 1 pixel separation between components. This is necessary in
  # case of densely labeled volumes in order for the erosion morphological
  # operation to work.
  #
  # An alternative would be to apply erosion separately to every object and then
  # compose the results together to form a single volume. This would incur a
  # a significant overhead compared to the approach used below.
  if is_2d:
    where = (eroded[:, :-1] != eroded[:, 1:]) & (eroded[:, :-1] != 0)
    eroded[:, 1:][where] = 0
    where = (eroded[:-1, :] != eroded[1:, :]) & (eroded[:-1, :] != 0)
    eroded[1:, :][where] = 0
  else:
    where = (eroded[:, :, :-1] != eroded[:, :, 1:]) & (eroded[:, :, :-1] != 0)
    eroded[:, :, 1:][where] = 0
    where = (eroded[:, :-1, :] != eroded[:, 1:, :]) & (eroded[:, :-1, :] != 0)
    eroded[:, 1:, :][where] = 0
    where = (eroded[:-1, :, :] != eroded[1:, :, :]) & (eroded[:-1, :, :] != 0)
    eroded[1:, :, :][where] = 0

  if radius > 0:
    if is_2d:
      struct = skimage.morphology.disk(radius)
    else:
      struct = skimage.morphology.ball(radius)
    eroded = skimage.morphology.erosion(eroded, footprint=struct)

  # Preserve small components.
  mask = np.in1d(labels.flat, small_segments).reshape(labels.shape)
  eroded[mask] = labels[mask]
  return eroded


def watershed_expand(seg: np.ndarray,
                     voxel_size: Sequence[float],
                     max_distance: Optional[float] = None,
                     mask: Optional[np.ndarray] = None):
  """Grows existing segments using watershed.

  All segments are grown at an uniform rate, using the Euclidean distance
  transform of the empty space of the input segmentation. This results in
  all empty voxels getting assigned the ID of the nearest segment, up to
  `max_distance`.

  Args:
    seg: 3d int ZYX array of segmentation data
    voxel_size: x, y, z voxel size in nm
    max_distance: max distance in nm to expand the seeds
    mask: 3d bool array of the same shape as `seg`; positive values define the
      region where watershed will be applied. If not specified, wastershed is
      applied everywhere in the subvolume.

  Returns:
    expanded segmentation, distance transform over the empty space
    of the original segmentation prior to expansion
  """
  # Map to low IDs for watershed to work.
  seg_low, orig_to_low = make_contiguous(seg)
  dist_map = edt.edt(seg_low == 0, anisotropy=voxel_size[::-1])

  if mask is None:
    mask = np.ones(seg_low.shape, dtype=bool)

  if max_distance is not None:
    mask[dist_map > max_distance] = False

  ws = skimage.segmentation.watershed(
      dist_map, seg_low, mask=mask).astype(np.uint64)

  # Restore any segment parts that might have been removed by the mask.
  nmask = np.logical_not(mask)
  if np.any(nmask):
    ws[nmask] = seg_low[nmask]

  orig_ids, low_ids = zip(*orig_to_low)
  return relabel(ws, np.array(low_ids), np.array(orig_ids)), dist_map

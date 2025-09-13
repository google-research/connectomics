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
from typing import AbstractSet, Iterable, Optional, Sequence

import edt
import networkx as nx
import numpy as np
import skimage.measure
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
  mask = np.isin(labels.flat, small_segments).reshape(labels.shape)
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


def split_disconnected_components(labels: np.ndarray, connectivity=1):
  """Relabels the connected components of a 3-D integer array.

  Connected components are determined based on 6-connectivity, where two
  neighboring positions are considering part of the same component if they have
  identical labels.

  The label 0 is treated specially: all positions labeled 0 in the input are
  labeled 0 in the output, regardless of whether they are contiguous.

  Connected components of the input array (other than segment id 0) are given
  consecutive ids in the output, starting from 1.

  Args:
    labels: 3-D integer numpy array.
    connectivity: 1, 2, or 3; for 6-, 18-, or 26-connectivity respectively.

  Returns:
    The relabeled numpy array, same dtype as `labels`.
  """
  has_zero = 0 in labels
  fixed_labels = skimage.measure.label(
      labels, connectivity=connectivity, background=0)
  if has_zero or (not has_zero and 0 in fixed_labels):
    if np.any((fixed_labels == 0) != (labels == 0)):
      fixed_labels[...] += 1
      fixed_labels[labels == 0] = 0
  return np.asarray(fixed_labels, dtype=labels.dtype)


def get_border_ids(vol3d: np.ndarray, inplane: bool = False) -> set[int]:
  """Finds ids of objects adjacent to the border of a 3d subvolume."""
  ret = (set(np.unique(vol3d[:, 0, :]))  #
         | set(np.unique(vol3d[:, -1, :]))  #
         | set(np.unique(vol3d[:, :, 0]))  #
         | set(np.unique(vol3d[:, :, -1])))
  if not inplane:
    ret |= set(np.unique(vol3d[0, :, :])) | set(np.unique(vol3d[-1, :, :]))
  return ret


def merge_internal_objects(bcc: list[AbstractSet[int]], aps: AbstractSet[int],
                           todo_bcc_idx: Iterable[int]) -> dict[int, int]:
  """Merges objects that are completely internal to other objects.

  Takes as input biconnected components (BCCs) and articulation points (APs)
  of a region adjacency graph (RAG) representing a segmentation.

  Args:
    bcc: list of sets of nodes of BCCs of the RAG
    aps: set of APs of the RAG
    todo_bcc_idx: indices in `bcc` for components that should be considered for
      merging

  Returns:
    map from BCC index to new label for the BCC
  """
  ap_to_bcc_idx = {}  # AP -> indices of BCCs they are a part of
  for ap in aps:
    ap_to_bcc_idx[ap] = {i for i, cc in enumerate(bcc) if ap in cc}

  ap_merge_forest = nx.DiGraph()
  to_merge = []

  while True:
    start_len = len(to_merge)
    remaining_bccs = []
    for cc_i in todo_bcc_idx:
      cc = bcc[cc_i]
      cc_aps = set(cc & aps)

      if len(cc_aps) == 1:
        # Direct merge of the BCC into the only AP that is part of it.
        to_merge.append(cc_i)
        cc_ap = cc_aps.pop()
        ap_to_bcc_idx[cc_ap].remove(cc_i)
      elif len([cc_ap for cc_ap in cc_aps if len(ap_to_bcc_idx[cc_ap]) > 1
               ]) == 1:
        # Merge into an AP that is the only remaining AP that is part of
        # more than the current BCC.
        to_merge.append(cc_i)
        target = None
        for cc_ap in cc_aps:
          if len(ap_to_bcc_idx[cc_ap]) > 1:
            target = cc_ap
          ap_to_bcc_idx[cc_ap].remove(cc_i)

        assert target is not None
        for cc_ap in cc_aps:
          if cc_ap == target:
            continue
          ap_merge_forest.add_edge(target, cc_ap)
      else:
        # The current BCC cannot be merged in this iteration because it
        # still contains multiple APs that are part of more than 1 BCC.
        remaining_bccs.append(cc_i)

    todo_bcc_idx = remaining_bccs

    # Terminate if no merges were applied in the last iteration.
    if len(to_merge) == start_len:
      break

  # Build the AP relabel map by exploring the AP merge forest starting
  # from the target labels (roots).
  ap_relabel = {}
  roots = [n for n, deg in ap_merge_forest.in_degree if deg == 0]
  for root in roots:
    for n in nx.dfs_preorder_nodes(ap_merge_forest, source=root):
      ap_relabel[n] = root

  bcc_relabel = {}
  for bcc_i in to_merge:
    cc = bcc[bcc_i]
    adjacent_aps = cc & aps

    targets = set([ap_relabel.get(ap, ap) for ap in adjacent_aps])
    assert len(targets) == 1
    target = targets.pop()

    bcc_relabel[bcc_i] = target

  return bcc_relabel


def create_boundary_mask_volume(
    shape: tuple[int, int, int],
    before_xyz: tuple[int, int, int] = (0, 0, 0),
    after_xyz: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
  """Creates a mask for the boundary of a volume.

  Args:
    shape: Shape of the volume.
    before_xyz: Voxels to set to False at the beginning of each dimension.
    after_xyz: Voxels to set to False at the end of each dimension.

  Returns:
    Boolean mask volume.
  """
  assert len(shape) == len(before_xyz) == len(after_xyz) == 3
  mask = np.ones(shape, dtype=bool)
  mask[:before_xyz[0], :, :] = False
  mask[:, :before_xyz[1], :] = False
  mask[:, :, :before_xyz[2]] = False
  if after_xyz[0] > 0:
    mask[-after_xyz[0]:, :, :] = False
  if after_xyz[1] > 0:
    mask[:, -after_xyz[1]:, :] = False
  if after_xyz[2] > 0:
    mask[:, :, -after_xyz[2]:] = False
  return mask

# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Subvolume processors for segmentation."""

import collections
from connectomics.common import graph_utils
from connectomics.segmentation import labels
from connectomics.segmentation import rag
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import numpy as np


class MergeInternalObjects(subvolume_processor.SubvolumeProcessor):
  """Merges internal objects into the containing ones.

  An object A is considered internal to a containing object B if all
  paths from A to the 'exterior' pass through B. The exterior is
  defined as the set of objects that touch the border of a subvolume.

  If the segmentation is represented as a region adjacency graph (RAG),
  with connected components of the background considered as labeled
  segmments, the above definition implies that the containing object
  is an articulation point (AP) in the RAG.
  """

  crop_at_borders = False

  def __init__(self, ignore_bg=False):
    """Constructor.

    Args:
      ignore_bg: if True, ignores the background component in the RAG
        construction; this causes objects only adjacent to one other
        ("containing") object and the background component to be considered
        "internal" and thus eligible for merging; use with caution.
    """
    super().__init__()
    self.ignore_bg = ignore_bg

  def process(self, subvol: subvolume.Subvolume) -> subvolume.SubvolumeOrMany:
    box = subvol.bbox
    input_ndarray = subvol.data
    with subvolume_processor.timer_counter('segmentation-prep'):
      seg3d = input_ndarray[0, ...]
      if self.ignore_bg:
        no_bg = seg3d.copy()
      else:
        no_bg = seg3d + 1
      no_bg_ccs = labels.split_disconnected_components(no_bg)
      border_ids = labels.get_border_ids(no_bg_ccs)

      cc_ids, indices = np.unique(no_bg_ccs, return_index=True)
      orig_ids = seg3d.ravel()[indices]
      cc_to_orig = dict(zip(cc_ids, orig_ids))

      g = rag.from_subvolume(no_bg_ccs)
      if self.ignore_bg and 0 in g:
        g.remove_node(0)
        border_ids.discard(0)

      if not g:  # Should only occur on empty input.
        return self.crop_box_and_data(box, input_ndarray)

    with subvolume_processor.timer_counter('ap-graph'):
      bcc, aps = graph_utils.biconnected_dfs(g, start_points=border_ids)

    with subvolume_processor.timer_counter('define-relabel'):
      # Any biconnected component (BCC) of the graph containing objects which
      # are not target (labeled) APs and which touch the border, cannot be
      # collapsed in the merging process.
      no_zero_aps = frozenset(n for n in aps if cc_to_orig[n] != 0)
      border_bcc_idx = {
          i for i, cc in enumerate(bcc) if (cc - no_zero_aps) & border_ids
      }
      mergeable_bcc_idx = set(range(len(bcc))) - border_bcc_idx

      # Find BBCs that can be collapsed and their new labels.
      bcc_relabel = labels.merge_internal_objects(bcc, aps, mergeable_bcc_idx)
      relabel = {}
      for bcc_i, label in bcc_relabel.items():
        cc = bcc[bcc_i]
        for n in cc:
          relabel[n] = label

          if cc_to_orig[n] == 0:
            subvolume_processor.counter('merged-segments-zero').inc()
          else:
            subvolume_processor.counter('merged-segments-nonzero').inc()

    with subvolume_processor.timer_counter('apply-relabel'):
      if relabel:
        for i, cc_id in enumerate(cc_ids):
          if cc_id in relabel:
            orig_ids[i] = cc_to_orig[relabel[cc_id]]

        # Map back to original ID space and perform mergers.
        ret = labels.relabel(no_bg_ccs, cc_ids, orig_ids)
      else:
        ret = seg3d

    subvolume_processor.counter('subvolumes-done').inc()
    return self.crop_box_and_data(box, ret[np.newaxis, ...])


class FillHoles(subvolume_processor.SubvolumeProcessor):
  """Fills holes in segments.

  A hole is a connected component of the background segment (0)
  that touches exactly one non-background segment and does not
  touch the border of the subvolume, both assuming 6-connectivity
  along the canonical axes.

  Run with context ~ 2x largest expected hole diameter to avoid
  edge effects.
  """

  crop_at_borders = False

  def __init__(self, min_neighbor_size=1000, inplane=False):
    """Constructor.

    Args:
      min_neighbor_size: minimum size (in voxels) of an object within the
        current subvolume for it to be considered a neighboring segment;
        settings this to small non-zero value allows filing of empty space
        completely embedded in large segments when this space also contains
        small (< specified size) labeled components
      inplane: whether to treat the segmentation as 2d and fill holes within XY
        planes
    """
    super().__init__()
    self._min_neighbor_size = min_neighbor_size
    self._inplane = inplane

  def _fill_holes(self, seg3d):
    """Fills holes in a segmentation subvolumes."""

    no_bg = seg3d + 1
    no_bg_ccs = labels.split_disconnected_components(no_bg)
    sizes = dict(zip(*np.unique(no_bg_ccs, return_counts=True)))
    border = labels.get_border_ids(no_bg_ccs, inplane=self._inplane)

    # Any connected component that used to be background that does not touch
    # the border of the volume is potentially a hole to be filled.
    hole_labels = set(np.unique(no_bg_ccs[no_bg == 1])) - border
    subvolume_processor.counter('potential-holes').inc(len(hole_labels))
    hole_mask = np.isin(no_bg_ccs, list(hole_labels))

    # (a, b) pairs where 'a' is background and 'b' is labeled.
    # Looks for neighboring segments assuming 6-connectivity.
    seg_nbor_pairs = set()
    for dim in 0, 1, 2:
      sel_offset = [slice(None)] * 3
      sel_offset[dim] = np.s_[:-1]

      sel_base = [slice(None)] * 3
      sel_base[dim] = np.s_[1:]

      sel_offset = tuple(sel_offset)
      sel_base = tuple(sel_base)

      # Right neighbor; 'b' is to the right of 'a'.
      right_bg = hole_mask[sel_offset]
      seg_nbor_pairs |= set(zip(no_bg_ccs[sel_offset][right_bg].ravel(),
                                no_bg_ccs[sel_base][right_bg].ravel()))

      # Left neighbor, 'b' is to the left of 'a'.
      left_bg = hole_mask[sel_base]
      seg_nbor_pairs |= set(zip(no_bg_ccs[sel_base][left_bg].ravel(),
                                no_bg_ccs[sel_offset][left_bg].ravel()))

    cc_ids, indices = np.unique(no_bg_ccs, return_index=True)
    orig_ids = seg3d.ravel()[indices]
    cc_to_orig = dict(zip(cc_ids, orig_ids))

    # Maps connected components of the background region to adjacent
    # segments.
    bg_to_nbors = collections.defaultdict(set)
    for a, b in seg_nbor_pairs:
      if sizes[b] >= self._min_neighbor_size:
        bg_to_nbors[a].add(cc_to_orig[b])

    # Build a relabel map mapping hole IDs to the IDs of the segments
    # containing them.
    relabel = {}
    for bg, nbors in bg_to_nbors.items():
      nbors.discard(0)
      # If there is more than 1 neighboring labeled component, this is
      # not a hole.
      if len(nbors) != 1:
        continue
      relabel[bg] = nbors.pop()
      subvolume_processor.counter('holes-filled').inc()

    if not relabel:
      return seg3d

    for i, cc_id in enumerate(cc_ids):
      if cc_id in relabel:
        assert orig_ids[i] == 0
        orig_ids[i] = relabel[cc_id]

    # Fill holes and map IDs back to the original ID space.
    return labels.relabel(no_bg_ccs, cc_ids, orig_ids)

  def process(self, subvol: subvolume.Subvolume) -> subvolume.SubvolumeOrMany:
    box = subvol.bbox
    input_ndarray = subvol.data
    seg3d = input_ndarray[0, ...]
    if self._inplane:
      ret = np.zeros_like(seg3d)
      for z in range(seg3d.shape[0]):
        ret[z : z + 1, ...] = self._fill_holes(seg3d[z : z + 1, ...])
    else:
      ret = self._fill_holes(seg3d)

    return self.crop_box_and_data(box, ret[np.newaxis, ...])

# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Utilities for region adjacency graphs (RAGs)."""

import networkx as nx
import numpy as np
from scipy import spatial


def from_subvolume(vol3d: np.ndarray) -> nx.Graph:
  """Returns the RAG for a 3d subvolume.

  Uses 6-connectvity to find neighbors. Only works for segmentations
  with IDs that fit in a uint32.

  Args:
    vol3d: 3d ndarray with the segmentation

  Returns:
    the corresponding RAG
  """
  assert np.max(vol3d) < 2**32

  # Looks for neighboring segments assuming 6-connectivity.
  seg_nbor_pairs = set()
  for dim in 0, 1, 2:
    sel_offset = [slice(None)] * 3
    sel_offset[dim] = np.s_[:-1]

    sel_base = [slice(None)] * 3
    sel_base[dim] = np.s_[1:]

    a = vol3d[tuple(sel_offset)].ravel()
    b = vol3d[tuple(sel_base)].ravel()
    x = a | (b << 32)
    x = x[a != b]
    unique_joint_labels = np.unique(x)

    seg_nbor_pairs |= set(
        zip(unique_joint_labels & 0xFFFFFFFF, unique_joint_labels >> 32))

  g = nx.Graph()
  g.add_edges_from(seg_nbor_pairs)
  return g


def from_set(kdts: dict[int, spatial._ckdtree.cKDTree]) -> nx.Graph:
  """Builds a RAG for a set of segments relying on their spatial proximity.

  A typical use case is to transform an equivalence set into a graph using
  skeleton or other point-based representation of segments.

  Args:
    kdts: map from segment IDs to k-d trees of associated spatial coordinates

  Returns:
    adjacency graph with greedily chosen edges connecting the most proximal
    segment pairs
  """
  segment_ids = list(kdts.keys())
  pairs = {}
  # For every pair of segments, identify the closest point pair.
  for i in range(len(segment_ids)):
    for j in range(i + 1, len(segment_ids)):
      dist, idx = kdts[segment_ids[i]].query(kdts[segment_ids[j]].data, k=1)
      ii = np.argmin(dist)
      pairs[(segment_ids[i], segment_ids[j])] = (dist[ii], idx[ii], ii)

  dists = [(v[0], v[1], v[2], k) for k, v in pairs.items()]
  dists.sort()

  uf = nx.utils.UnionFind()
  g = nx.Graph()

  for dist, idx1, idx2, (id1, id2) in dists:
    if uf[id1] == uf[id2]:
      continue

    uf.union(id1, id2)
    g.add_edge(id1, id2, idx={id1: idx1, id2: idx2})

  return g

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


cKDTree = spatial._ckdtree.cKDTree  # pylint:disable=protected-access


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

  g = nx.Graph()
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

    seg_nbor_pairs = set(
        zip(unique_joint_labels & 0xFFFFFFFF, unique_joint_labels >> 32)
    )
    g.add_edges_from(seg_nbor_pairs, dim=dim)

  return g


def _graph_from_pairs(
    g: nx.Graph,
    pairs: dict[tuple[int, int], tuple[float, int, int]],
) -> nx.Graph:
  """Builds a RAG from a set of segment pairs.

  Args:
    g: initial RAG
    pairs: map from segment ID pairs to tuples of (distance, index1, index2)

  Returns:
    adjacency graph with greedily chosen edges connecting the most proximal
    segment pairs
  """
  dists = [(dist, idx1, idx2, k) for k, (dist, idx1, idx2), in pairs.items()]
  dists.sort()

  uf = nx.utils.UnionFind()

  for dist, idx1, idx2, (id1, id2) in dists:
    if uf[id1] == uf[id2]:
      continue

    uf.union(id1, id2)
    g.add_edge(id1, id2, idx={id1: idx1, id2: idx2})

  return g


def _connect_components(g: nx.Graph, kdts: dict[int, cKDTree]) -> nx.Graph:
  """Ensures that the graph is fully connected.

  Connects separate components greedily based on maximal proximity.

  Args:
    g: initial graph defining how segments are connected
    kdts: map from segment IDs to k-d trees of associated spatial coordinates

  Returns:
    graph with all components connected
  """

  if nx.number_connected_components(g) <= 1:
    return g

  # Builds a KD-tree for each connected component.
  ccs = list(nx.connected_components(g))
  cc_kdts = {}
  cc_to_seg = {}
  cc_to_idx = {}
  for i, cc in enumerate(ccs):
    points = []
    seg_ids = []
    idxs = []
    for seg_id in cc:
      kdt = kdts[seg_id]
      points.extend(kdt.data)
      seg_ids.extend([seg_id] * len(kdt.data))
      idxs.extend(list(range(len(kdt.data))))

    cc_kdts[i] = cKDTree(np.array(points))
    cc_to_seg[i] = seg_ids
    cc_to_idx[i] = idxs

  cc_g = from_set(cc_kdts)
  for cc_i, cc_j, data in cc_g.edges(data=True):
    id_to_idx = data['idx']
    idx_i = id_to_idx[cc_i]
    idx_j = id_to_idx[cc_j]
    id1 = cc_to_seg[cc_i][idx_i]
    id2 = cc_to_seg[cc_j][idx_j]
    g.add_edge(
        id1, id2, idx={id1: cc_to_idx[cc_i][idx_i], id2: cc_to_idx[cc_j][idx_j]}
    )

  assert nx.number_connected_components(g) <= 1
  return g


def from_set(kdts: dict[int, cKDTree]) -> nx.Graph:
  """Builds a RAG for a set of segments relying on their spatial proximity.

  A typical use case is to transform an equivalence set into a graph using
  skeleton or other point-based representation of segments. This has O(N^2)
  complexity.

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
      pairs[(segment_ids[i], segment_ids[j])] = (
          dist[ii],
          int(idx[ii]),
          int(ii),
      )

  g = nx.Graph()
  g.add_nodes_from(segment_ids)

  return _graph_from_pairs(g, pairs)


def from_set_nn(kdts: dict[int, cKDTree], max_dist: float) -> nx.Graph:
  """Like 'from_set', but uses a more efficient two-stage procedure.

  First, a local neighborhood search is performed O(N log N), followed
  by O(n^2) reconnection of 'n' connected components if necessary.

  Args:
    kdts: map from segments IDs to k-d trees of associated spatial coordinates
    max_dist: maximum distance within which to search for neighbors (typical
      distance between segments) in physical units

  Returns:
    adjacency graph with greedily chosen edges connecting the most proximal
    segment pairs
  """
  all_points = []
  point_to_seg = []
  point_to_idx = []

  for seg_id, kdt in kdts.items():
    all_points.extend(list(kdt.data))
    point_to_seg.extend([seg_id] * len(kdt.data))
    point_to_idx.extend(list(range(len(kdt.data))))

  g = nx.Graph()
  g.add_nodes_from(list(kdts.keys()))

  if not point_to_seg:
    return g

  all_points = np.array(all_points)
  combined_kdt = cKDTree(all_points)

  # Find nearest neighbors within the radius for each point.
  pairs = {}
  for i, point in enumerate(all_points):
    nbor_indices = combined_kdt.query_ball_point(point, max_dist)
    for j in nbor_indices:
      if i >= j:
        continue

      seg_i = point_to_seg[i]
      seg_j = point_to_seg[j]

      if seg_i != seg_j:
        pair = seg_i, seg_j
        dist = np.linalg.norm(point - all_points[j])

        if pair not in pairs or dist < pairs[pair][0]:
          pairs[pair] = (dist, point_to_idx[i], point_to_idx[j])

  g = _graph_from_pairs(g, pairs)
  g = _connect_components(g, kdts)
  return g

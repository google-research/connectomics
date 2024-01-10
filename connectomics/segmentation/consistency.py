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
"""Library for computing skeleton consistency and optimal cuts."""
import collections
from typing import Optional, Sequence
import networkx as nx
import numpy as np

Node = int
Edge = tuple[Node, Node]


class NodeCounter:
  """ABC for counters used by CentripetalSkeletonConsistency."""

  def total_counts(self) -> np.ndarray:
    raise NotImplementedError

  def add_node(self, node: Node, counts: np.ndarray):
    raise NotImplementedError


class IndexCounter(NodeCounter):
  """Counter where node label is used to index into the counter vector."""

  def __init__(self, nx_skeleton: nx.Graph, node_property_name: str):
    self._nx_skeleton = nx_skeleton
    self._node_property_name = node_property_name

  def total_counts(self) -> np.ndarray:
    """Count node labels for entire nx_skeleton.

    Returns:
      ndarray of counts with length equal to max label + 1.
    """
    node_labels = [
        v for _, v in self._nx_skeleton.nodes(data=self._node_property_name)
    ]
    labels, counts = np.unique(node_labels, return_counts=True)
    total_counts = np.zeros(max(labels) + 1, dtype=np.int64)
    for label, count in zip(labels, counts):
      total_counts[label] = count
    return total_counts

  def add_node(self, node: Node, counts: np.ndarray):
    """Increment count for label of given node."""
    label = self._nx_skeleton.nodes[node][self._node_property_name]
    counts[label] += 1


class VectorCounter(NodeCounter):
  """Counter for nodes that hold vector counts / probabilities already."""

  def __init__(self, nx_skeleton: nx.Graph, node_property_name: str):
    self._nx_skeleton = nx_skeleton
    self._node_property_name = node_property_name

  def total_counts(self) -> np.ndarray:
    node_counts_or_probabilities = [
        v for _, v in self._nx_skeleton.nodes(data=self._node_property_name)
    ]
    return np.sum(node_counts_or_probabilities, axis=0)

  def add_node(self, node: Node, counts: np.ndarray):
    counts += self._nx_skeleton.nodes[node][self._node_property_name]


class UnitCounter(NodeCounter):
  """Counter where every node just gets a value of 1.

  This is used for just counting leaving nodes rather than actually computing
  consistencies.
  """

  def __init__(self, nx_skeleton: nx.Graph):
    self._nx_skeleton = nx_skeleton

  def total_counts(self) -> np.ndarray:
    return np.array([len(self._nx_skeleton)])

  def add_node(self, unused_node: Node, counts: np.ndarray):
    counts[0] += 1


class CentripetalSkeletonConsistency(object):
  """Visits edges from leaf nodes in, tallying the consistency for each cut."""

  def __init__(self, nx_skeleton: nx.Graph, counter: NodeCounter,
               remain_sources: Sequence[Node] = ()):
    """Constructor.

    Args:
      nx_skeleton: NetworkX skeleton whose edges will be annotated in place.
      counter: NodeCounter to use to tally node labels / counts / probabilities.
      remain_sources: A Sequence of Nodes; if given, we will start the cut
        search as far away from these sources as possible.  This makes it likely
        that the orientation of cuts will have leaving direction away from
        remain_sources.

    Raises:
      ValueError: if input nx_skeleton is not a single connected component.
    """
    if nx.number_connected_components(nx_skeleton) != 1:
      raise ValueError(
          'Skeleton consistency only works with single connected component.')
    self._nx_skeleton = nx_skeleton
    self._counter = counter
    self._total_label_counts = self._counter.total_counts()
    self._remain_sources = remain_sources

  def _consistency(self, class_label_counts: np.ndarray) -> float:
    if class_label_counts.size == 0:
      return 0.0
    return float(class_label_counts.max())

  def init_consistency(self):
    """Get the initial global consistency of nx_skeleton."""
    return self._consistency(self._total_label_counts)

  def _find_leaf_nodes(self) -> list[Node]:
    """Find leaf nodes as BFS from any remain_sources.

    Returns:
      List of leaf nodes, in order visited by BFS starting from remain_sources.
      This order is important so that the cut search can be biased to generate
      cuts with orientation such that the leaving direction points away from
      remain_sources.
    """
    nx_skeleton = self._nx_skeleton
    to_visit = collections.deque(self._remain_sources)
    if not to_visit:  # No remain_sources; start anywhere.
      to_visit.append(next(iter(nx_skeleton)))

    visited = set()
    leaf_nodes = []
    while to_visit:
      node = to_visit.popleft()
      visited.add(node)
      if nx_skeleton.degree(node) == 1:
        leaf_nodes.append(node)
      unvisited_neighbors = set(nx_skeleton.neighbors(node)) - visited
      to_visit.extend(unvisited_neighbors)
    return leaf_nodes

  def _normalize_edge(self, edge: Edge) -> Edge:
    n0, n1 = edge
    return (n0, n1) if n0 < n1 else (n1, n0)

  def best_consistency_cut(self,
                           filter_func=None) -> tuple[Optional[Edge], float]:
    """Moves from leaf nodes in, annotating edges and tracking best cut.

    Annotates 'leaving_counts' and 'leaving_direction_node' in each edge as it
    moves in.  The leaving_counts are then used to compute the consistency of
    the 'leaving' branch and the 'remaining' branch, and thus the global post-
    cut consistency.

    The algorithm starts at leaf nodes and moves in until it reaches a branch
    point.  We cannot move in from branch points until all adjacent edges but
    one are visited.  The leaf nodes are used in reverse order, and traversal is
    by DFS; this makes it likely that leaving directions are oriented away from
    any remain_sources.

    Args:
      filter_func: Optional filter function accepting leaving_counts,
          cut_consistency, and init_consistency.  If given, cuts for which
          filter function returns False will not be considered for best cut.

    Returns:
      (best_cut, best_cut_consistency)
      best_cut: (node0, node1) edge identifying the best consistency cut, or
          None if no good cut is found.
      best_cut_consistency: float giving the consistency of the best cut, or
          init_consistency if no good cut is found.

    Raises:
      networkx.HasACycle: if the algorithm detects a cycle due to failure to
    visit all edges.
      ValueError: if the algorithm fails to visit all edges without detecting
    a cycle.  This can occur if the graph has been previously traversed and
    leaving_counts are already marked on edges.
    """
    init_consistency = self.init_consistency()
    best_cut = None, init_consistency
    start_nodes = self._find_leaf_nodes()
    if not start_nodes:
      return best_cut

    edges_visited = 0
    while start_nodes:
      start_node = start_nodes.pop()
      edges = self._nx_skeleton.edges(nbunch=start_node, data='leaving_counts')
      unvisited_edges = [edge for edge in edges if edge[2] is None]
      if not unvisited_edges:
        continue  # This node is done.
      if len(unvisited_edges) > 1:
        # If we have more than one unvisited edge out, then we can't move in
        # from here at this time.  Wait for this node to be added back into
        # start_nodes later.
        continue
      edge_to_visit = unvisited_edges[0][:2]

      # Check whether cutting here is the new best.
      summed_counts = np.zeros_like(self._total_label_counts)
      for edge in edges:
        if edge[2] is not None:
          summed_counts += edge[2]
      self._counter.add_node(start_node, summed_counts)
      remain_counts = self._total_label_counts - summed_counts
      leave_consistency = self._consistency(summed_counts)
      remain_consistency = self._consistency(remain_counts)
      consistency = leave_consistency + remain_consistency
      if consistency > best_cut[1]:
        if filter_func is None or filter_func(
            leaving_counts=summed_counts, cut_consistency=consistency,
            init_consistency=init_consistency):
          best_cut = self._normalize_edge(edge_to_visit), consistency

      # Mark this one visited and add the next node.
      data = self._nx_skeleton.edges[edge_to_visit]
      data['leaving_counts'] = summed_counts
      data['leaving_direction_node'] = start_node  # Not used here, but useful.
      n0, n1 = edge_to_visit
      start_nodes.append(n0 if n0 != start_node else n1)
      edges_visited += 1

    # If there are still unvisited edges, we might have gotten stuck at a cycle,
    # which prevents unvisted_edges == 1 above.
    if edges_visited < self._nx_skeleton.number_of_edges():
      if nx.cycle_basis(self._nx_skeleton):
        raise nx.HasACycle('Failed to visit all edges; cycle detected.')
      raise ValueError('Failed to visit all edges; graph previously traversed?')
    return best_cut

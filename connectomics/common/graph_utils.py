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
"""Utilities for manipulating graphs."""

import itertools
import networkx as nx


def _rindex(seq, item):
  """Returns the 1st position of `item` in `seq`, searching from the end."""
  for i, v in enumerate(reversed(seq)):
    if item == v:
      return len(seq) - i - 1
  raise ValueError('%r not in list' % item)


# In contrast to networkx/algorithms/components/biconnected, this uses rindex
# for ~20x faster execution. In the networkx implementation, computing BCCs
# is 55x slower than computing APs for a 256^3 segmentation subvolume RAG,
# due to repeated slow linear searches over a potentially large edge
# stack. Also returns both APs and BBCs from a single pass.
def biconnected_dfs(
    g: nx.Graph,
    start_points=None) -> tuple[list[frozenset[int]], frozenset[int]]:
  """Returns the biconnected components and articulation points of a graph."""
  visited = set()
  aps = set()
  bccs = []

  if start_points is None:
    start_points = []

  for start in itertools.chain(start_points, g):
    if start in visited:
      continue
    discovery = {start: 0}  # time of first discovery of node during search
    low = {start: 0}
    root_children = 0
    visited.add(start)
    edge_stack = []
    stack = [(start, start, iter(g[start]))]

    while stack:
      grandparent, parent, children = stack[-1]
      try:
        child = next(children)
        if grandparent == child:
          continue
        if child in visited:
          if discovery[child] <= discovery[parent]:  # back edge
            low[parent] = min(low[parent], discovery[child])
            # Record edge, but don't follow.
            edge_stack.append((parent, child))
        else:
          low[child] = discovery[child] = len(discovery)
          visited.add(child)
          stack.append((parent, child, iter(g[child])))
          edge_stack.append((parent, child))
      except StopIteration:
        stack.pop()
        if len(stack) > 1:
          if low[parent] >= discovery[grandparent]:
            ind = _rindex(edge_stack, (grandparent, parent))
            bccs.append(
                frozenset(itertools.chain.from_iterable(edge_stack[ind:])))
            edge_stack = edge_stack[:ind]
            aps.add(grandparent)
          low[grandparent] = min(low[parent], low[grandparent])
        elif stack:  # length 1 so grandparent is root
          root_children += 1
          ind = _rindex(edge_stack, (grandparent, parent))
          bccs.append(
              frozenset(itertools.chain.from_iterable(edge_stack[ind:])))

    # Root node is articulation point if it has more than 1 child.
    if root_children > 1:
      aps.add(start)

  return bccs, frozenset(aps)

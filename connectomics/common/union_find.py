# Copyright 2024 The Google Research Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements the disjoint-set forest/union-find data structure."""


class UnionFind:
  """Dictionary-based implementation of the disjoint-set forest data structure.

  Uses union-by-rank and path compression. Works with any hashable type.
  Does not store singleton sets.

  More info: http://en.wikipedia.org/wiki/Disjoint-set_data_structure
  """

  def __init__(self):
    self._parents = {}
    self._ranks = {}

  def __bool__(self):
    return bool(self._parents)

  def Union(self, a, b):
    """Merge 'a' and 'b' into a single set."""
    root_a = self.Find(a)
    root_b = self.Find(b)
    if root_a == root_b:
      return

    rank_a = self._ranks.setdefault(root_a, 1)
    rank_b = self._ranks.setdefault(root_b, 1)

    if rank_a < rank_b:
      self._parents[root_a] = root_b
    elif rank_a > rank_b:
      self._parents[root_b] = root_a
    else:
      self._parents[root_b] = root_a
      self._ranks[root_a] += 1

  def IsSingleton(self, a):
    """Returns whether set 'a' only contains a single element."""
    return a not in self._ranks

  def Find(self, a):
    """Finds the representative of 'a'.

    If 'a' was not seen before, treats it as a singleton set.

    Args:
      a: object to find a representative for

    Returns:
      representative of 'a'
    """
    if a not in self._parents:
      return a

    # Find representative.
    path = [a]
    root = self._parents[a]
    while root != path[-1]:
      path.append(root)
      root = self._parents.get(root, root)

    # Compress path.
    for ancestor in path[::-1]:
      self._parents[ancestor] = root

    return root

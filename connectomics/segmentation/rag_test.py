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
"""Tests for rag."""

from absl.testing import absltest
from connectomics.segmentation import rag
import networkx as nx
import numpy as np
from scipy import spatial


class RagTest(absltest.TestCase):

  def test_from_set_points(self):
    # Each segment is associated with just a single 3d point.
    kdts = {
        1: spatial.cKDTree([(1, 1, 1)]),
        2: spatial.cKDTree([(2, 1, 1)]),
        3: spatial.cKDTree([(3, 1, 1)]),
    }
    g = rag.from_set(kdts)
    self.assertTrue(nx.utils.edges_equal(g.edges(), ((1, 2), (2, 3))))

    # All segments will be connected in the 1st (subquadratic) pass.
    g2 = rag.from_set_nn(kdts, max_dist=10)
    self.assertTrue(nx.utils.graphs_equal(g, g2))

    # All segments will be connected in the 2nd (quadratic) pass.
    g2 = rag.from_set_nn(kdts, max_dist=0.1)
    self.assertTrue(nx.utils.graphs_equal(g, g2))

  def test_from_set_skeletons(self):
    # Each segment is associated with a short skeleton fragment.
    skels = {
        1: nx.Graph([(0, 1), (1, 2), (2, 3)]),
        2: nx.Graph([(0, 1), (1, 2), (2, 3)]),
        3: nx.Graph([(0, 1), (1, 2)]),
    }

    # Add spatial coordinates for all skeleton nodes.
    skels[1].nodes[0]['position'] = (0, 1, 0)
    skels[1].nodes[1]['position'] = (0, 2, 0)
    skels[1].nodes[2]['position'] = (0, 3, 0)
    skels[1].nodes[3]['position'] = (0, 4, 0)  # *

    skels[2].nodes[0]['position'] = (0, 5, 1)  # *
    skels[2].nodes[1]['position'] = (0, 5, 2)
    skels[2].nodes[2]['position'] = (0, 5, 3)
    skels[2].nodes[3]['position'] = (0, 5, 4)  # %

    skels[3].nodes[0]['position'] = (0, 8, 6)
    skels[3].nodes[1]['position'] = (0, 7, 5)
    skels[3].nodes[2]['position'] = (0, 6, 4)  # %

    # Convert skeletons to k-d trees and build RAG.
    kdts = {
        k: spatial.cKDTree([n['position'] for _, n in v.nodes(data=True)])
        for k, v in skels.items()
    }
    g = rag.from_set(kdts)
    self.assertTrue(nx.utils.edges_equal(g.edges(), ((1, 2), (2, 3))))

    # Verify which specific points got connected (marked with * and %
    # in the comments above).
    self.assertEqual(g.edges[1, 2]['idx'][1], 3)
    self.assertEqual(g.edges[1, 2]['idx'][2], 0)

    self.assertEqual(g.edges[2, 3]['idx'][2], 3)
    self.assertEqual(g.edges[2, 3]['idx'][3], 2)

    # All segments will be connected in the 1st (subquadratic) pass.
    g2 = rag.from_set_nn(kdts, max_dist=10)
    self.assertTrue(nx.utils.graphs_equal(g, g2))

    # All segments will be connected in the 2nd (quadratic) pass.
    g2 = rag.from_set_nn(kdts, max_dist=0.1)
    self.assertTrue(nx.utils.graphs_equal(g, g2))

  def test_from_subvolume(self):
    seg = np.zeros((10, 10, 2), dtype=np.uint64)
    seg[2:, :, 0] = 1
    seg[1:, 3:4, 1] = 3
    seg[1:, 5:6, 1] = 2
    seg[2:, 7:, 1] = 3

    result = rag.from_subvolume(seg)
    expected = nx.Graph()
    expected.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (0, 3)])

    self.assertTrue(nx.is_isomorphic(result, expected))


if __name__ == '__main__':
  absltest.main()

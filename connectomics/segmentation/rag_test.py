# coding=utf-8
# Copyright 2023-2026 The Google Research Authors.
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
from absl.testing import parameterized
from connectomics.segmentation import rag
import networkx as nx
import numpy as np
from scipy import spatial


class RagTest(parameterized.TestCase):

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

  @parameterized.parameters(
      np.uint8, np.uint16, np.uint32, np.uint64, np.int32, np.int64
  )
  def test_from_subvolume_dtypes(self, dtype):
    seg = np.zeros((10, 10, 2), dtype=dtype)
    seg[2:, :, 0] = 1
    seg[1:, 3:4, 1] = 3
    seg[1:, 5:6, 1] = 2
    seg[2:, 7:, 1] = 3

    result = rag.from_subvolume(seg)
    expected = {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    }
    actual = {(min(u, v), max(u, v)) for u, v in result.edges()}
    self.assertSetEqual(actual, expected)

  def test_from_subvolume_axis_dim_and_non_contiguous_views(self):
    # Distinct adjacent pairs along each axis only:
    # dim 0 -> (11, 12), dim 1 -> (21, 22), dim 2 -> (31, 32).
    seg = np.zeros((4, 4, 4), dtype=np.uint32)
    seg[0, 0, 0] = 11
    seg[1, 0, 0] = 12
    seg[0, 2, 0] = 21
    seg[0, 3, 0] = 22
    seg[2, 0, 2] = 31
    seg[2, 0, 3] = 32

    g = rag.from_subvolume(seg)
    self.assertEqual(g.edges[11, 12]['dim'], 0)
    self.assertEqual(g.edges[21, 22]['dim'], 1)
    self.assertEqual(g.edges[31, 32]['dim'], 2)

    # Non-contiguous transpose swaps axis 0 and axis 2.
    g_t = rag.from_subvolume(seg.T)
    self.assertEqual(g_t.edges[31, 32]['dim'], 0)
    self.assertEqual(g_t.edges[21, 22]['dim'], 1)
    self.assertEqual(g_t.edges[11, 12]['dim'], 2)

    # Strided non-contiguous view matches contiguous copy.
    padded = np.repeat(seg, 2, axis=1)
    g_strided = rag.from_subvolume(padded[:, ::2, :])
    self.assertTrue(nx.utils.graphs_equal(g, g_strided))

  def test_from_subvolume_empty_and_uniform(self):
    empty = np.zeros((0, 4, 4), dtype=np.uint32)
    self.assertEqual(rag.from_subvolume(empty).number_of_edges(), 0)

    uniform = np.full((4, 4, 4), 7, dtype=np.uint16)
    self.assertEqual(rag.from_subvolume(uniform).number_of_edges(), 0)

  def test_from_subvolume_rejects_out_of_range_ids(self):
    too_large = np.zeros((2, 2, 2), dtype=np.uint64)
    too_large[0, 0, 0] = 1 << 32
    with self.assertRaises(AssertionError):
      rag.from_subvolume(too_large)

    negative = np.zeros((2, 2, 2), dtype=np.int32)
    negative[0, 0, 0] = -1
    with self.assertRaises(AssertionError):
      rag.from_subvolume(negative)

  def test_from_subvolume_rejects_invalid_ndim_or_dtype(self):
    with self.assertRaises(AssertionError):
      rag.from_subvolume(np.zeros((1, 2, 2, 2), dtype=np.uint32))

    with self.assertRaises(AssertionError):
      rag.from_subvolume(np.zeros((2, 2, 2), dtype=np.float32))


if __name__ == '__main__':
  absltest.main()

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
"""Tests for connectomics.segmentation.consistency."""

from absl.testing import absltest
from connectomics.segmentation import consistency
import networkx as nx
import numpy as np


class CountersTest(absltest.TestCase):

  def test_index_counter(self):
    path10 = nx.generators.path_graph(10)
    for i in range(10):
      path10.nodes[i]['class_label'] = i // 2
    index_counter = consistency.IndexCounter(path10, 'class_label')
    total_counts = index_counter.total_counts()
    np.testing.assert_equal(total_counts, np.ones(5) * 2)
    self.assertEqual(total_counts.dtype, np.int64)

    counts = np.zeros(5, dtype=np.int64)
    expected = counts.copy()
    expected[2] = 1
    index_counter.add_node(5, counts)
    np.testing.assert_equal(counts, expected)

  def test_vector_counter(self):
    path10 = nx.generators.path_graph(10)
    for i in range(10):
      path10.nodes[i]['class_probabilities'] = [0.9, 0.1]
    vector_counter = consistency.VectorCounter(path10, 'class_probabilities')
    total_counts = vector_counter.total_counts()
    np.testing.assert_almost_equal(total_counts, np.array([9.0, 1.0]))


class ConsistencyTest(absltest.TestCase):

  def test_class_label_best_consistency_cut(self):
    path10 = nx.generators.path_graph(10)
    for i in range(5):
      path10.nodes[i]['class_label'] = 1
    for i in range(5, 10):
      path10.nodes[i]['class_label'] = 2
    # Use remain_sources=[9] to cause cut search to start from node 0.
    csc = consistency.CentripetalSkeletonConsistency(
        path10, consistency.IndexCounter(path10, 'class_label'),
        remain_sources=[9])
    self.assertEqual(csc.init_consistency(), 5.0)
    best_cut, best_cut_consistency = csc.best_consistency_cut()

    self.assertCountEqual(best_cut, (4, 5))
    self.assertEqual(best_cut_consistency, 10.0)
    np.testing.assert_equal(path10.edges[(1, 2)]['leaving_counts'],
                            np.array([0, 2, 0]))
    self.assertEqual(path10.edges[(1, 2)]['leaving_direction_node'], 1)

  def test_class_probability_best_consistency_cut(self):
    path10 = nx.generators.path_graph(10)
    for i in range(5):
      path10.nodes[i]['class_probability'] = 0.9, 0.1
    for i in range(5, 10):
      path10.nodes[i]['class_probability'] = 0.1, 0.9
    csc = consistency.CentripetalSkeletonConsistency(
        path10, consistency.VectorCounter(path10, 'class_probability'))
    self.assertAlmostEqual(csc.init_consistency(), 5.0)
    best_cut, best_cut_consistency = csc.best_consistency_cut()
    self.assertCountEqual(best_cut, (4, 5))
    self.assertAlmostEqual(best_cut_consistency, 9.0)


if __name__ == '__main__':
  absltest.main()

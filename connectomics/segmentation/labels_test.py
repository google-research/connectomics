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
"""Tests for labels."""

from absl.testing import absltest
from connectomics.segmentation import labels
import numpy as np


class SplitDisconnectedComponentsTest(absltest.TestCase):

  def test_disconnected_components(self):
    old_labels = np.array([[[2, 2, 3, 4],
                            [2, 3, 3, 2],
                            [0, 0, 0, 2],
                            [0, 3, 3, 3]]], dtype=np.uint64)
    new_labels = labels.split_disconnected_components(old_labels)
    expected = np.array([[[1, 1, 2, 3],
                          [1, 2, 2, 4],
                          [0, 0, 0, 4],
                          [0, 5, 5, 5]]], dtype=np.uint64)
    self.assertTrue(labels.are_equivalent(new_labels, expected))


class UtilsTest(absltest.TestCase):

  def test_equivalence(self):
    a = np.array
    self.assertTrue(
        labels.are_equivalent(a([1, 1, 2, 3]), a([2, 2, 3, 4])))
    self.assertFalse(
        labels.are_equivalent(a([1, 2, 2, 3]), a([2, 2, 3, 4])))
    self.assertFalse(
        labels.are_equivalent(a([1, 1, 2, 3]), a([1, 2, 3, 4])))

  def test_make_contiguous(self):
    original = (np.random.random((50, 50)) * 1e16).astype(np.int64)
    original[49, 49] = 0
    original[0, 0] = int(1e18)
    compacted, _ = labels.make_contiguous(original)
    self.assertEqual(np.max(compacted), np.unique(original).size - 1)
    self.assertTrue(labels.are_equivalent(original, compacted))

    # Should preserve the 0 label, as this often has special meaning
    # (background).
    self.assertEqual(0, compacted[49, 49])

  def test_relabel(self):
    orig_ids = np.array([2 << 40, 100000, 30], dtype=np.uint64)
    new_ids = [1, 2, 3]
    original = np.random.choice(orig_ids, size=100)
    relabeled = labels.relabel(original, orig_ids, new_ids)

    self.assertTrue(np.all(relabeled[original == (2 << 60)] == 1))
    self.assertTrue(np.all(relabeled[original == 100000] == 2))
    self.assertTrue(np.all(relabeled[original == 30] == 3))

  def test_erode(self):
    ids = np.zeros((20, 20), dtype=np.uint64)
    ids[:10, :] = 1
    ids[10:, :] = 2
    eroded = labels.erode(ids, min_size=0, radius=1)

    expected = ids.copy()
    expected[9:12] = 0
    np.testing.assert_equal(expected, eroded)

  def test_watershed_expand(self):
    seg = np.random.randint(0, 10, size=(100, 100, 100), dtype=np.uint64)
    eroded = labels.erode(seg, radius=3)

    # Ignore any objects that were completely removed during erosion.
    large_enough = np.logical_not(np.isin(seg.ravel(),
                                          np.unique(eroded))).reshape(seg.shape)
    expanded, _ = labels.watershed_expand(seg, voxel_size=(1, 1, 1))
    np.testing.assert_array_equal(expanded[large_enough], seg[large_enough])


if __name__ == '__main__':
  absltest.main()

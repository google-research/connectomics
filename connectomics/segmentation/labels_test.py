# coding=utf-8
# Copyright 2022-2026 The Google Research Authors.
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
from absl.testing import parameterized
from connectomics.segmentation import labels
import numpy as np


class SplitDisconnectedComponentsTest(parameterized.TestCase):

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
    self.assertEqual(new_labels.dtype, np.uint64)
    self.assertTrue(labels.are_equivalent(new_labels, expected))
    np.testing.assert_array_equal(new_labels == 0, old_labels == 0)

  @parameterized.parameters(
      np.uint8, np.uint16, np.uint32, np.uint64, np.int32, np.int64
  )
  def test_preserves_dtype_and_background(self, dtype):
    vol = np.zeros((4, 4, 4), dtype=dtype)
    vol[0, 0, :2] = 7
    vol[2, 2, :2] = 7
    vol[3, 3, 3] = 9

    out = labels.split_disconnected_components(vol)
    self.assertEqual(out.dtype, dtype)
    np.testing.assert_array_equal(out == 0, vol == 0)
    self.assertSequenceEqual(sorted(np.unique(out).tolist()), [0, 1, 2, 3])
    self.assertNotEqual(int(out[0, 0, 0]), int(out[2, 2, 0]))

  def test_dense_volume_without_background(self):
    vol = np.ones((3, 3, 4), dtype=np.uint64)
    vol[:, :, 2:] = 2
    vol[0, 0, 0] = 3

    out = labels.split_disconnected_components(vol)
    self.assertEqual(out.dtype, np.uint64)
    self.assertFalse(np.any(out == 0))
    self.assertSequenceEqual(sorted(np.unique(out).tolist()), [1, 2, 3])

  def test_all_zero_volume(self):
    vol = np.zeros((3, 4, 5), dtype=np.uint64)
    out = labels.split_disconnected_components(vol)
    self.assertEqual(out.dtype, np.uint64)
    np.testing.assert_array_equal(out, vol)

  def test_connectivity_modes(self):
    vol = np.zeros((3, 3, 3), dtype=np.uint32)
    vol[0, 0, 0] = 5
    vol[1, 1, 0] = 5
    vol[2, 2, 1] = 5

    cc6 = labels.split_disconnected_components(vol, connectivity=1)
    cc18 = labels.split_disconnected_components(vol, connectivity=2)
    cc26 = labels.split_disconnected_components(vol, connectivity=3)

    self.assertEqual(int(cc6.max()), 3)
    self.assertEqual(int(cc18.max()), 2)
    self.assertEqual(int(cc26.max()), 1)


class UtilsTest(parameterized.TestCase):

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

  def test_relabel_sparse_large_ids(self):
    rng = np.random.default_rng(0)
    orig_ids = np.array([2 << 40, 100000, 30], dtype=np.uint64)
    new_ids = np.array([1, 2, 3], dtype=np.uint64)
    original = rng.choice(orig_ids, size=200)
    original[0] = 2 << 40
    original[1] = 100000
    original[2] = 30
    relabeled = labels.relabel(original, orig_ids, new_ids)

    self.assertEqual(relabeled.dtype, np.uint64)
    np.testing.assert_array_equal(relabeled[original == (2 << 40)], 1)
    np.testing.assert_array_equal(relabeled[original == 100000], 2)
    np.testing.assert_array_equal(relabeled[original == 30], 3)

  @parameterized.parameters(
      np.uint8, np.uint16, np.uint32, np.uint64, np.int32, np.int64
  )
  def test_relabel_dense_lut_fast_path(self, dtype):
    orig_ids = np.array([50, 0, 12, 7], dtype=dtype)
    new_ids = np.array([5, 99, 0, 42], dtype=np.uint16)
    vol = np.array(
        [[[0, 50, 12], [7, 0, 50]], [[12, 7, 0], [50, 12, 7]]], dtype=dtype
    )
    expected = np.array(
        [[[99, 5, 0], [42, 99, 5]], [[0, 42, 99], [5, 0, 42]]],
        dtype=np.uint16,
    )

    relabeled = labels.relabel(vol, orig_ids, new_ids)
    self.assertEqual(relabeled.dtype, np.uint16)
    np.testing.assert_array_equal(relabeled, expected)

    # Verify non-contiguous view handling.
    strided = vol[:, :, ::2]
    relabeled_strided = labels.relabel(strided, orig_ids, new_ids)
    np.testing.assert_array_equal(relabeled_strided, expected[:, :, ::2])

  def test_relabel_lut_matches_fallback_boundary_and_signed(self):
    # Compare dense LUT path (max_id == 9_999_999) against searchsorted fallback
    # (max_id == 10_000_000) and negative signed IDs.
    orig_lut = np.array([0, 42, 9_999_999], dtype=np.int64)
    orig_fallback = np.array([0, 42, 10_000_000], dtype=np.int64)
    new_ids = np.array([0, 7, 8], dtype=np.int64)

    vol_lut = np.array([[0, 42, 9_999_999], [9_999_999, 0, 42]], dtype=np.int64)
    vol_fb = np.array(
        [[0, 42, 10_000_000], [10_000_000, 0, 42]], dtype=np.int64
    )
    np.testing.assert_array_equal(
        labels.relabel(vol_lut, orig_lut, new_ids),
        labels.relabel(vol_fb, orig_fallback, new_ids),
    )

    neg_vol = np.array([[-5, 0, 12], [12, -5, 0]], dtype=np.int32)
    neg_out = labels.relabel(neg_vol, [12, -5, 0], [1, 2, 0])
    np.testing.assert_array_equal(
        neg_out, np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int64)
    )

    empty = np.zeros((0, 3, 3), dtype=np.uint32)
    empty_out = labels.relabel(empty, np.array([0], dtype=np.uint32), [0])
    self.assertEqual(empty_out.shape, (0, 3, 3))

    with self.assertRaises(IndexError):
      labels.relabel(
          np.array([0, 99], dtype=np.uint32),
          np.array([0, 42], dtype=np.uint32),
          np.array([0, 7], dtype=np.uint32),
      )

  @parameterized.parameters(
      (1 << 64) - 1,
      (1 << 64) - 43,
      1 << 63,
  )
  def test_relabel_rejects_out_of_range_high_uint64_ids(self, bad_id):
    orig_ids = np.array([0, 42], dtype=np.uint64)
    new_ids = np.array([0, 7], dtype=np.uint64)
    vol = np.array([[0, 42, 0], [42, bad_id, 0]], dtype=np.uint64)

    with self.assertRaises(IndexError):
      labels.relabel(vol, orig_ids, new_ids)

    with self.assertRaises(IndexError):
      labels.relabel(vol[:, 1::2], orig_ids, new_ids)

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

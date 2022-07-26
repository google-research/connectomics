# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Tests for subvolume."""

from absl.testing import absltest
from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.volume import subvolume
import numpy as np
import numpy.testing as npt

BBox = bounding_box.BoundingBox
Subvol = subvolume.Subvolume


class SubvolumeTest(absltest.TestCase):

  def test_check_bbox_dims(self):
    data = np.ones([1, 12, 11, 10])
    subvolume._check_bbox_dims(BBox([0, 0, 0], [10, 11, 12]), data)
    subvolume._check_bbox_dims(BBox([110, 120, 130], [10, 11, 12]), data)
    with self.assertRaises(ValueError):
      subvolume._check_bbox_dims(BBox([0, 0, 0], [10, 11, 13]), data)

  def test_construction(self):
    data = np.ones([1, 12, 11, 10])
    bbox = BBox([0, 0, 0], [10, 11, 12])
    sv = Subvol(data, bbox)
    self.assertTrue(sv.valid)
    self.assertEqual((1, 12, 11, 10), sv.shape)
    npt.assert_array_equal(bbox.size, sv.bbox.size)
    npt.assert_array_equal(sv.bbox.size, sv.size)
    npt.assert_array_equal(bbox.start, sv.bbox.start)
    npt.assert_array_equal(bbox.start, sv.start)

    with self.assertRaises(ValueError):
      _ = Subvol(data)
    with self.assertRaises(ValueError):
      _ = Subvol(sv, bbox)
    self.assertEqual(sv, Subvol(sv))

  def test_indexing(self):
    bbox = BBox([100, 200, 300], [20, 50, 100])
    data = np.zeros(bbox.size[::-1])
    data[0] = 1
    data = np.cumsum(
        np.cumsum(np.cumsum(data, axis=0), axis=1), axis=2, dtype=np.uint64)
    data = np.concatenate((data[np.newaxis],) * 4, axis=0)

    sv = Subvol(data, bbox)
    expected = data[:, 80:88, 13:17, 2:]

    # Test relative indexing
    indexed = sv[:, 80:88, 13:17, 2:]
    npt.assert_array_equal(expected, indexed.data)
    npt.assert_array_equal(BBox([102, 213, 380], [18, 4, 8]), indexed.bbox)

    # Test absolute indexing (values offset by the start of the bounding box).
    abs_indexed = sv.index_abs[:, 380:388, 213:217, 102:]
    npt.assert_array_equal(expected, abs_indexed.data)
    npt.assert_array_equal(BBox([102, 213, 380], [18, 4, 8]), abs_indexed.bbox)

    # Test integer indexing
    ind = np.s_[:, 8, 13:17, 2:]
    indexed = sv[ind]
    expected = data[array.normalize_index(ind, data.shape)]
    npt.assert_array_equal(expected, indexed.data)

    ind = np.s_[2, 8, 1, 2:]
    indexed = sv[ind]
    expected = data[array.normalize_index(ind, data.shape)]
    npt.assert_array_equal(expected, indexed.data)

    ind = np.s_[2:3, 8, 1, 2:]
    indexed = sv[ind]
    expected = data[array.normalize_index(ind, data.shape)]
    npt.assert_array_equal(expected, indexed.data)

  def test_new_bounding_box(self):
    data = np.ones([1, 12, 11, 10])
    bbox = BBox([0, 0, 0], [10, 11, 12])
    new_bbox = bbox.translate([100, 100, 100])

    sv = Subvol(data, bbox)
    sv.new_bounding_box(new_bbox)
    npt.assert_array_equal(sv.bbox.start, new_bbox.start)

    with self.assertRaises(ValueError):
      sv.new_bounding_box(BBox([1, 2, 3], [4, 5, 6]))

  def test_clip(self):
    data = np.ones([1, 12, 11, 10])
    bbox = BBox([0, 0, 0], [10, 11, 12])

    sub_bbox = BBox([1, 2, 3], [4, 5, 6])

    sv = Subvol(data, bbox)
    clipped_sv = sv.clip(sub_bbox)

    self.assertEqual((1, 2, 3), tuple(clipped_sv.bbox.start))
    self.assertEqual((4, 5, 6), tuple(clipped_sv.bbox.size))
    self.assertEqual((5, 7, 9), tuple(clipped_sv.bbox.end))
    self.assertTrue(clipped_sv.valid)

    npt.assert_array_equal(data[sub_bbox.to_slice4d()], clipped_sv.data)

  def test_clip_no_intersection(self):
    data = np.ones([1, 12, 11, 10])
    bbox = BBox([0, 0, 0], [10, 11, 12])

    sub_bbox = BBox([19, 29, 39], [4, 5, 6])

    sv = Subvol(data, bbox)
    clipped_sv = sv.clip(sub_bbox)
    self.assertEqual((10, 11, 12), tuple(clipped_sv.bbox.start))
    self.assertEqual((0, 0, 0), tuple(clipped_sv.bbox.size))
    self.assertEqual((10, 11, 12), tuple(clipped_sv.bbox.end))
    self.assertEqual((1, 0, 0, 0), clipped_sv.shape)
    self.assertFalse(clipped_sv.valid)
    self.assertEqual(0, clipped_sv.data.size)

  def test_merge_with(self):
    zeros = np.zeros([1, 10, 10, 10])
    ones = np.ones([1, 10, 10, 10])

    bbox = BBox([0, 0, 0], [10, 10, 10])
    a = Subvol(zeros.copy(), bbox)
    b = Subvol(ones.copy(), bbox)

    npt.assert_array_equal(a.data, zeros)
    npt.assert_array_equal(b.data, ones)

    npt.assert_array_equal(a.merge_with(b).data, ones)
    npt.assert_array_equal(a.data, ones)
    npt.assert_array_equal(b.data, ones)

    # Check no values to fill
    npt.assert_array_equal(a.merge_with(ones + 1).data, ones)

    # Check nonzero values to fill
    npt.assert_array_equal(a.merge_with(ones + 1, empty_value=1).data, ones + 1)
    npt.assert_array_equal(a.data, ones + 1)

    # Check NaN fill values
    slices = np.s_[0, 4:7, 1:4, :]
    a.data[slices] = np.nan
    self.assertTrue(np.all(np.isnan(a.data[slices])))

    nan_mask = np.isnan(a.data)
    a.merge_with(zeros, empty_value=np.nan)
    self.assertFalse(np.any(np.isnan(a.data)))
    npt.assert_array_equal(a.data[nan_mask], zeros[nan_mask])
    inan_mask = np.logical_not(nan_mask)
    npt.assert_array_equal(a.data[inan_mask], ones[inan_mask] * 2)

    with self.assertRaises(ValueError):
      a.merge_with(np.ones([11, 11, 11]))

  def test_merge_with_offset_subvol(self):
    zeros = np.zeros([1, 10, 10, 10])
    ones = np.ones([1, 10, 10, 10])

    bbox = BBox([0, 0, 0], [10, 10, 10])
    a = Subvol(zeros.copy(), bbox)
    b = Subvol(ones.copy(), bbox.translate([3, 3, 3]))

    a.merge_with(b)
    npt.assert_array_equal(a.data[:, 3:, 3:, 3:].ravel(), 1)
    npt.assert_array_equal(a.data[:, :3, :, :].ravel(), 0)
    npt.assert_array_equal(a.data[:, :, :3, :].ravel(), 0)
    npt.assert_array_equal(a.data[:, :, :, :3].ravel(), 0)

  def test_slice_partial_overlap(self):
    sv = subvolume.Subvolume(
        np.ones([1, 10, 10, 10]), BBox([35, 35, 35], [10, 10, 10]))
    prior_corner_bbox = BBox([30, 30, 30], [10, 10, 10])
    clipped_sv = sv.index_abs[prior_corner_bbox.to_slice4d()]
    self.assertEqual(clipped_sv.bbox, BBox([35, 35, 35], [5, 5, 5]))

  def test_split(self):
    bbox = BBox([100, 200, 300], [10, 20, 30])
    data = np.zeros(bbox.size[::-1])
    data[0] = 1
    data = np.cumsum(
        np.cumsum(np.cumsum(data, axis=0), axis=1), axis=2, dtype=np.uint64)
    data = np.concatenate((data[np.newaxis],) * 4, axis=0)

    sv = Subvol(data, bbox)

    splits = sv.split(np.array([10, 10, 10]), origin=np.array([0, 0, 0]))
    self.assertLen(splits, 6)
    self.assertTrue(all([np.all(s.bbox.size == [10, 10, 10]) for s in splits]))

    expected_corners = [
        [100, 200, 300],
        [100, 200, 310],
        [100, 210, 300],
        [100, 210, 310],
        [100, 200, 320],
        [100, 210, 320],
    ]
    split_corners = [s.bbox.start for s in splits]
    for corner in expected_corners:
      self.assertTrue(any([np.all(corner == s) for s in split_corners]))

  def test_split_unaligned(self):
    bbox = BBox([101, 202, 303], [10, 20, 30])
    data = np.zeros(bbox.size[::-1])
    data[0] = 1
    data = np.cumsum(
        np.cumsum(np.cumsum(data, axis=0), axis=1), axis=2, dtype=np.uint64)
    data = np.concatenate((data[np.newaxis],) * 4, axis=0)

    sv = Subvol(data, bbox)

    splits = sv.split(np.array([10, 10, 10]), origin=np.array([70, 70, 70]))
    self.assertLen(splits, 24)
    self.assertTrue(all([np.all(s.bbox.size == [10, 10, 10]) for s in splits]))
    expected_corners = [
        [100, 200, 300],
        [110, 200, 300],
        [100, 210, 300],
        [110, 210, 300],
        [100, 220, 300],
        [110, 220, 300],
        [100, 200, 310],
        [110, 200, 310],
        [100, 210, 310],
        [110, 210, 310],
        [100, 220, 310],
        [110, 220, 310],
        [100, 200, 320],
        [110, 200, 320],
        [100, 210, 320],
        [110, 210, 320],
        [100, 220, 320],
        [110, 220, 320],
        [100, 200, 330],
        [110, 200, 330],
        [100, 210, 330],
        [110, 210, 330],
        [100, 220, 330],
        [110, 220, 330],
    ]
    split_corners = [s.bbox.start for s in splits]
    for corner in expected_corners:
      self.assertTrue(any([np.all(corner == s) for s in split_corners]))

  def test_split_no_origin(self):
    bbox = BBox([101, 202, 303], [10, 20, 30])
    data = np.zeros(bbox.size[::-1])
    data[0] = 1
    data = np.cumsum(
        np.cumsum(np.cumsum(data, axis=0), axis=1), axis=2, dtype=np.uint64)
    data = np.concatenate((data[np.newaxis],) * 4, axis=0)

    sv = Subvol(data, bbox)

    splits = sv.split(np.array([10, 10, 10]))
    self.assertLen(splits, 6)
    self.assertTrue(all([np.all(s.bbox.size == [10, 10, 10]) for s in splits]))

    expected_corners = [
        [101, 202, 303],
        [101, 202, 313],
        [101, 212, 303],
        [101, 212, 313],
        [101, 202, 323],
        [101, 212, 323],
    ]
    split_corners = [s.bbox.start for s in splits]
    for corner in expected_corners:
      self.assertTrue(any([np.all(corner == s) for s in split_corners]))

  def test_split_and_clip(self):
    bbox = BBox([103, 204, 305], [10, 20, 30])
    data = np.zeros(bbox.size[::-1])
    data[0] = 1
    data = np.cumsum(
        np.cumsum(np.cumsum(data, axis=0), axis=1), axis=2, dtype=np.uint64)
    data = np.concatenate((data[np.newaxis],) * 4, axis=0)

    sv = Subvol(data, bbox)

    splits = sv.split(
        np.array([10, 10, 10]),
        origin=np.array([0, 0, 0]),
        clip_output_subvolumes=True)
    self.assertLen(splits, 24)
    # Sort ensures return order doesn't matter
    starts = np.sort(np.array([s.bbox.start for s in splits]), axis=0)
    expected_starts = [
        [103, 204, 305],
        [103, 204, 305],
        [103, 204, 305],
        [103, 204, 305],
        [103, 204, 305],
        [103, 204, 305],
        [103, 204, 310],
        [103, 204, 310],
        [103, 210, 310],
        [103, 210, 310],
        [103, 210, 310],
        [103, 210, 310],
        [110, 210, 320],
        [110, 210, 320],
        [110, 210, 320],
        [110, 210, 320],
        [110, 220, 320],
        [110, 220, 320],
        [110, 220, 330],
        [110, 220, 330],
        [110, 220, 330],
        [110, 220, 330],
        [110, 220, 330],
        [110, 220, 330],
    ]
    npt.assert_array_equal(starts, expected_starts)
    sizes = np.sort(np.array([s.bbox.size for s in splits]), axis=0)
    expected_sizes = [
        [3, 4, 5],
        [3, 4, 5],
        [3, 4, 5],
        [3, 4, 5],
        [3, 4, 5],
        [3, 4, 5],
        [3, 4, 5],
        [3, 4, 5],
        [3, 6, 5],
        [3, 6, 5],
        [3, 6, 5],
        [3, 6, 5],
        [7, 6, 10],
        [7, 6, 10],
        [7, 6, 10],
        [7, 6, 10],
        [7, 10, 10],
        [7, 10, 10],
        [7, 10, 10],
        [7, 10, 10],
        [7, 10, 10],
        [7, 10, 10],
        [7, 10, 10],
        [7, 10, 10],
    ]
    npt.assert_array_equal(sizes, expected_sizes)


if __name__ == '__main__':
  absltest.main()

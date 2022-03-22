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
"""Tests for connectomics.common.BoundingBox."""

from absl.testing import absltest
from connectomics.common import bounding_box
import numpy as np

Box = bounding_box.BoundingBox
FloatBox = bounding_box.FloatBoundingBox


class BoundingBoxTest(absltest.TestCase):

  def test_construction_start_end(self):
    b = Box(start=[1, 2, 3], end=[4, 5, 6])
    self.assertEqual(3, b.rank)
    self.assertEqual([1, 2, 3], list(b.start))
    self.assertEqual([4, 5, 6], list(b.end))
    self.assertEqual([3, 3, 3], list(b.size))

  def test_assign(self):
    b = Box(start=[1, 2, 3], end=[4, 5, 6])
    self.assertEqual(1, b.start[0])

  def test_construction_start_size(self):
    b = Box(start=[1, 2, 3], size=[3, 3, 3])
    self.assertEqual(3, b.rank)
    self.assertEqual([1, 2, 3], list(b.start))
    self.assertEqual([4, 5, 6], list(b.end))
    self.assertEqual([3, 3, 3], list(b.size))

  def test_construction_start_size_broadcast(self):
    b = Box(start=[1, 2, 3], size=3)
    self.assertEqual(3, b.rank)
    self.assertEqual([1, 2, 3], list(b.start))
    self.assertEqual([4, 5, 6], list(b.end))
    self.assertEqual([3, 3, 3], list(b.size))

  def test_construction_end_size(self):
    b = Box(end=[4, 5, 6], size=[3, 3, 3])
    self.assertEqual(3, b.rank)
    self.assertEqual([1, 2, 3], list(b.start))
    self.assertEqual([4, 5, 6], list(b.end))
    self.assertEqual([3, 3, 3], list(b.size))

  def test_construction_end_size_broadcast(self):
    b = Box(end=(4, 5, 6), size=3)
    self.assertEqual(3, b.rank)
    self.assertEqual([1, 2, 3], list(b.start))
    self.assertEqual([4, 5, 6], list(b.end))
    self.assertEqual([3, 3, 3], list(b.size))

  def test_construction_errors(self):
    with self.assertRaises(ValueError):
      Box(start=1, end=2)
    with self.assertRaises(ValueError):
      Box(start=1, size=2)
    with self.assertRaises(ValueError):
      Box(end=2, size=1)
    with self.assertRaises(ValueError):
      Box(start=(1, 2), end=(3, 4, 5))
    with self.assertRaises(ValueError):
      Box(start=(1, 2), size=(3, 4, 5))
    with self.assertRaises(ValueError):
      Box(end=(1, 2), size=(3, 4, 5))

  def test_attribute_setting(self):
    b = Box(start=(0, 1, 2), end=(3, 4, 5))
    np.testing.assert_array_equal((3, 3, 3), b.size)
    np.testing.assert_array_equal((0, 1, 2), b.start)
    np.testing.assert_array_equal((3, 4, 5), b.end)

    with self.assertRaises(AttributeError):
      b.start = (2, 3, 4)
    with self.assertRaises(AttributeError):
      b.end = (2, 3, 4)
    with self.assertRaises(AttributeError):
      b.size = (2, 3, 4)

  def test_attribute_update(self):
    b = Box(start=(0, 1, 2), end=(3, 4, 5))
    with self.assertRaises(ValueError):
      b.start[:2] //= 2

    with self.assertRaises(ValueError):
      b.size[:2] //= 2

  def test_eq(self):
    self.assertEqual(True, (Box(start=[1, 2, 3], size=[4, 5, 6]) == Box(
        start=[1, 2, 3], size=[4, 5, 6])))
    self.assertEqual(False, (Box(start=[1, 2, 3], size=[4, 5, 6]) == Box(
        start=[1, 2, 3], size=[4, 5, 7])))
    self.assertEqual(False, (Box(start=[1, 2, 4], size=[4, 5, 6]) == Box(
        start=[1, 2, 3], size=[4, 5, 6])))
    self.assertEqual(True, (Box(start=[1, 2, 3], size=[4, 5, 6]) != Box(
        start=[1, 2, 3], size=[4, 5, 7])))

  def test_scale(self):
    self.assertEqual(
        Box(start=[4, 4, 4], size=[8, 8, 8]),
        Box(start=[2, 2, 2], size=[4, 4, 4]).scale(2))
    self.assertEqual(
        Box(start=[2, 4, 6], size=[4, 8, 12]),
        Box(start=[2, 2, 2], size=[4, 4, 4]).scale((1, 2, 3)))
    self.assertEqual(
        Box(start=[1, 1, 1], size=[2, 2, 2]),
        Box(start=[2, 2, 2], size=[4, 4, 4]).scale(.5))
    self.assertEqual(
        Box(start=[1, 1, 1], size=[2, 2, 2]),
        Box(start=[3, 3, 3], size=[3, 3, 3]).scale(.51))
    self.assertEqual(
        Box(start=[1, 1, 3], size=[2, 2, 3]),
        Box(start=[3, 3, 3], size=[3, 3, 3]).scale((.5, .5, 1)))
    self.assertEqual(
        Box(start=[4, 1, 4], size=[10, 4, 8]),
        Box(start=[2, 2, 2], size=[4, 4, 4]).scale((2.4, .9, 2)))

  def test_intersection(self):
    box0 = Box(start=[0, 1, 2], size=[3, 3, 3])
    box1 = Box(start=[2, 2, 3], size=[3, 3, 3])
    intersection = box0.intersection(box1)
    self.assertEqual([2, 2, 3], list(intersection.start))
    self.assertEqual([1, 2, 2], list(intersection.size))

    box2 = Box(start=[0, 1, 2], size=[3, 3, 1])
    intersection = box2.intersection(box1)
    self.assertIsNone(intersection)

    # Real world example.
    allen03_incep3_inference = Box(start=[47, 47, 1], size=[1021, 1021, 55])
    allen03_truth = Box(start=[344, 289, 0], size=[500, 500, 30])
    intersection = allen03_incep3_inference.intersection(allen03_truth)
    self.assertEqual([344, 289, 1], list(intersection.start))
    self.assertEqual([500, 500, 29], list(intersection.size))

  def test_hull(self):
    box0 = Box(start=[0, 1, 2], size=[3, 3, 3])
    box1 = Box(start=[2, 2, 1], size=[3, 3, 4])
    hull = box0.hull(box1)
    self.assertEqual(Box(start=[0, 1, 1], end=[5, 5, 5]), hull)

  def test_translate(self):
    box0 = Box(start=[0, 1, 2], size=[3, 3, 3])
    self.assertEqual(Box(start=[3, 4, 5], size=[3, 3, 3]), box0.translate(3))
    self.assertEqual(
        Box(start=[3, 3, 3], size=[3, 3, 3]), box0.translate([3, 2, 1]))

  def test_adjusted_by(self):
    box0 = Box(start=[0, 1, 2], size=[3, 3, 3])

    self.assertEqual(Box(start=[1, 2, 3], size=[2, 2, 2]), box0.adjusted_by(1))
    self.assertEqual(
        Box(start=[1, 2, 3], size=[2, 2, 2]), box0.adjusted_by([1, 1, 1]))

    self.assertEqual(
        Box(start=[0, 1, 2], size=[4, 4, 4]), box0.adjusted_by(end=1))
    self.assertEqual(
        Box(start=[0, 1, 2], size=[4, 4, 4]), box0.adjusted_by(end=[1, 1, 1]))

    self.assertEqual(
        Box(start=[1, 2, 3], size=[3, 3, 3]), box0.adjusted_by(start=1, end=1))
    self.assertEqual(
        Box(start=[1, 2, 3], size=[3, 3, 3]),
        box0.adjusted_by(start=(1, 1, 1), end=(1, 1, 1)))

  def test_relative(self):
    self.assertEqual(
        Box([3, 3, 3], [4, 5, 6]),
        Box([3, 3, 3], [4, 5, 6]).relative())

    self.assertEqual(
        Box([3, 3, 3], [2, 4, 6]),
        Box([1, 2, 3], [4, 5, 6]).relative(start=(2, 1, 0)))

    self.assertEqual(
        Box([1, 2, 3], end=[3, 3, 3]),
        Box([1, 2, 3], [4, 5, 6]).relative(end=(2, 1, 0)))

    self.assertEqual(
        Box([1, 3, 4], end=[3, 3, 7]),
        Box([1, 2, 3], [4, 5, 6]).relative(start=(0, 1, 1), end=(2, 1, 4)))

    self.assertEqual(
        Box([3, 3, 3], [3, 4, 5]),
        Box([1, 2, 3], [4, 5, 6]).relative(start=(2, 1, 0), size=(3, 4, 5)))

    self.assertEqual(
        Box(size=[3, 4, 5], end=[6, 6, 6]),
        Box([1, 2, 3], [4, 5, 6]).relative(end=(5, 4, 3), size=(3, 4, 5)))

  def test_intersections(self):
    box0 = Box([0, 1, 2], [3, 3, 3])
    box1 = Box([2, 2, 3], [3, 3, 3])
    box2 = Box([0, 1, 2], [3, 3, 1])

    intersections = bounding_box.intersections((box1,), (box0, box1, box2))
    self.assertLen(intersections, 2)
    self.assertEqual([2, 2, 3], intersections[0].start.tolist())
    self.assertEqual([1, 2, 2], intersections[0].size.tolist())
    np.testing.assert_array_equal(box1.start, intersections[1].start)
    np.testing.assert_array_equal(box1.size, intersections[1].size)

  def test_encompass(self):
    box0 = Box(start=[0, 1, 2], end=[3, 3, 3])
    self.assertEqual(box0, box0.encompass(box0))

    box1 = Box(start=[1, 2, 3], end=[5, 5, 5])
    self.assertEqual(Box(start=[0, 1, 2], end=[5, 5, 5]), box0.encompass(box1))

    box2 = Box(start=[1, 2, 2], end=[2, 2, 3])
    self.assertEqual(
        Box(start=[0, 1, 2], end=[5, 5, 5]), box0.encompass(box1, box2))

    box3 = Box(start=[10, 10, 10], size=[3, 4, 5])
    self.assertEqual(
        Box(start=[0, 1, 2], end=[13, 14, 15]),
        box0.encompass(box1, box2, box3))

  def test_to_slice(self):
    start = [0, 1, 2, 3, 4]
    end = [5, 6, 7, 8, 9]
    box = Box(start=start, end=end)

    expected = np.index_exp[start[4]:end[4], start[3]:end[3], start[2]:end[2],
                            start[1]:end[1], start[0]:end[0]]
    self.assertEqual(expected, box.to_slice_tuple())
    self.assertEqual(expected[1:], box.to_slice4d())
    self.assertEqual(expected[2:], box.to_slice3d())

    self.assertEqual(expected[:3], box.to_slice_tuple(2))
    self.assertEqual(expected[3:], box.to_slice_tuple(end_dim=2))
    self.assertEqual(expected[1:4], box.to_slice_tuple(start_dim=1, end_dim=4))
    self.assertEqual(expected[0:4], box.to_slice_tuple(start_dim=1, end_dim=5))
    self.assertEqual(expected[0:4], box.to_slice_tuple(start_dim=1, end_dim=7))
    self.assertEqual(tuple(), box.to_slice_tuple(start_dim=8, end_dim=9))
    self.assertEqual(tuple(), box.to_slice_tuple(start_dim=8, end_dim=4))

  def test_to_slice_float(self):
    start = [0.1, 1.1, 2.1, 3.1, 4.1]
    end = [5.1, 6.1, 7.1, 8.1, 9.1]
    box = FloatBox(start=start, end=end)

    expected = np.index_exp[start[4]:end[4], start[3]:end[3], start[2]:end[2],
                            start[1]:end[1], start[0]:end[0]]
    self.assertEqual(expected, box.to_slice_tuple())
    self.assertEqual(expected[1:], box.to_slice4d())
    self.assertEqual(expected[2:], box.to_slice3d())

    self.assertEqual(expected[:3], box.to_slice_tuple(2))
    self.assertEqual(expected[3:], box.to_slice_tuple(end_dim=2))
    self.assertEqual(expected[1:4], box.to_slice_tuple(start_dim=1, end_dim=4))
    self.assertEqual(expected[0:4], box.to_slice_tuple(start_dim=1, end_dim=5))
    self.assertEqual(expected[0:4], box.to_slice_tuple(start_dim=1, end_dim=7))
    self.assertEqual(tuple(), box.to_slice_tuple(start_dim=8, end_dim=9))
    self.assertEqual(tuple(), box.to_slice_tuple(start_dim=8, end_dim=4))

    # Test smaller-than-required dimensions
    start = [0, 1, 2]
    end = [5, 6, 7]
    expected = np.index_exp[start[2]:end[2], start[1]:end[1], start[0]:end[0]]
    none_slice = (slice(None, None, None),)
    for dim_size in range(len(start)):
      box = Box(start=start[:dim_size], end=end[:dim_size])

      expected_slice = expected[len(expected) - dim_size:]
      expected_3d = none_slice * (3 - dim_size) + expected_slice
      self.assertEqual(expected_3d, box.to_slice3d())
      expected_4d = none_slice * (4 - dim_size) + expected_slice
      self.assertEqual(expected_4d, box.to_slice4d())


class GlobalTest(absltest.TestCase):

  def test_intersections(self):
    box0 = Box(start=[0, 1, 2], end=[10, 10, 10])
    box1 = Box(start=[1, 2, 3], end=[5, 5, 5])

    self.assertCountEqual([
        Box(start=[1, 2, 3], end=[5, 5, 5]),
    ], bounding_box.intersections([box0], [box1]))

    self.assertCountEqual([
        Box(start=[1, 2, 3], end=[5, 5, 5]),
    ], bounding_box.intersections(box0, [box1]))
    self.assertCountEqual([
        Box(start=[1, 2, 3], end=[5, 5, 5]),
    ], bounding_box.intersections([box0], box1))

    box2 = Box(start=[3, 2, 3], end=[5, 5, 5])
    self.assertEqual([
        Box(start=[1, 2, 3], end=[5, 5, 5]),
        Box(start=[3, 2, 3], end=[5, 5, 5]),
    ], bounding_box.intersections([box0], [box1, box2]))
    self.assertEqual([
        Box(start=[1, 2, 3], end=[5, 5, 5]),
        Box(start=[3, 2, 3], end=[5, 5, 5]),
    ], bounding_box.intersections(box0, [box1, box2]))

    self.assertEqual([
        Box(start=[1, 2, 3], end=[5, 5, 5]),
        Box(start=[3, 2, 3], end=[5, 5, 5]),
    ], bounding_box.intersections([box0, box2], [box1]))
    self.assertEqual([
        Box(start=[1, 2, 3], end=[5, 5, 5]),
        Box(start=[3, 2, 3], end=[5, 5, 5]),
    ], bounding_box.intersections([box0, box2], box1))

    self.assertEqual([
        Box(start=[3, 2, 3], end=[5, 5, 5]),
        Box(start=[3, 2, 3], end=[5, 5, 5]),
    ], bounding_box.intersections([box0, box1], [box2]))

  def test_containing(self):
    with self.assertRaises(ValueError):
      bounding_box.containing()

    box0 = Box(start=[0, 1, 2], end=[10, 10, 10])
    box1 = Box(start=[1, 2, 3], end=[5, 5, 5])
    box2 = Box(start=[3, 2, 3], end=[15, 16, 17])
    box3 = Box(start=[6, 7, 8], end=[16, 17, 18])
    box4 = Box(start=[6, 7, 8], end=[17, 18, 19])

    self.assertEqual(
        Box(start=[0, 1, 2], end=[16, 17, 18]),
        bounding_box.containing(box0, box1, box2, box3))

    self.assertEqual(
        Box(start=[0, 1, 2], end=[17, 18, 19]),
        bounding_box.containing(box0, box1, box2, box3, box4))


if __name__ == '__main__':
  absltest.main()

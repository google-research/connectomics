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
"""Tests for connectomics.common.box_generator."""

from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.common import box_generator as m
import numpy as np

# Alias to make the test more concise
Box = bounding_box.BoundingBox
FloatBox = bounding_box.FloatBoundingBox


class BoxGeneratorTest(absltest.TestCase):

  def test_init(self):
    c = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(20, 20, 10)),
        box_size=(15, 13, 6),
        box_overlap=(1, 2, 2))

    self.assertEqual(Box(start=(0, 0, 0), size=(20, 20, 10)), c.outer_box)
    self.assertEqual(Box(start=(0, 0, 0), size=(2, 2, 2)), c.output)

    self.assertEqual(c.squeeze, [])
    self.assertEqual(c.num_boxes, 8)

  def test_init_float(self):
    c = m.BoxGenerator(
        FloatBox(start=(0.1, 0.1, 0.1), size=(20.1, 20.1, 10.1)),
        box_size=(15, 13, 6),
        box_overlap=(1, 2, 2))

    self.assertEqual(
        FloatBox(start=(0.1, 0.1, 0.1), size=(20.1, 20.1, 10.1)), c.outer_box)
    self.assertEqual(Box(start=(0, 0, 0), size=(2, 2, 3)), c.output)

    self.assertEqual(c.squeeze, [])
    self.assertEqual(c.num_boxes, 12)

  def test_init_single(self):
    c = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(20, 20, 10)), box_size=(20, 20, 10))

    self.assertEqual(Box(start=(0, 0, 0), size=(20, 20, 10)), c.outer_box)
    self.assertEqual(Box(start=(0, 0, 0), size=(1, 1, 1)), c.output)
    self.assertEqual(c.squeeze, [])
    self.assertEqual(c.num_boxes, 1)

  def test_generate(self):
    c = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(20, 20, 10)),
        box_size=(15, 13, 6),
        box_overlap=(1, 2, 2))

    self.assertEqual(c.num_boxes, 8)

    self.assertEqual(((0, 0, 0), Box(start=(0, 0, 0), end=(15, 13, 6))),
                     c.generate(0))
    self.assertEqual(((1, 0, 0), Box(start=(14, 0, 0), end=(20, 13, 6))),
                     c.generate(1))
    self.assertEqual(((0, 1, 0), Box(start=(0, 11, 0), end=(15, 20, 6))),
                     c.generate(2))
    self.assertEqual(((1, 1, 0), Box(start=(14, 11, 0), end=(20, 20, 6))),
                     c.generate(3))
    self.assertEqual(((0, 0, 1), Box(start=(0, 0, 4), end=(15, 13, 10))),
                     c.generate(4))
    self.assertEqual(((1, 0, 1), Box(start=(14, 0, 4), end=(20, 13, 10))),
                     c.generate(5))
    self.assertEqual(((0, 1, 1), Box(start=(0, 11, 4), end=(15, 20, 10))),
                     c.generate(6))
    self.assertEqual(((1, 1, 1), Box(start=(14, 11, 4), end=(20, 20, 10))),
                     c.generate(7))

    with self.assertRaises(ValueError):
      c.generate(8)

  def test_back_shift_small_boxes(self):
    c = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(20, 20, 10)),
        box_size=(15, 13, 6),
        box_overlap=(1, 2, 2),
        back_shift_small_boxes=True)

    self.assertEqual(c.num_boxes, 8)

    self.assertEqual(((0, 0, 0), Box(start=(0, 0, 0), end=(15, 13, 6))),
                     c.generate(0))
    self.assertEqual(((1, 0, 0), Box(start=(5, 0, 0), end=(20, 13, 6))),
                     c.generate(1))
    self.assertEqual(((0, 1, 0), Box(start=(0, 7, 0), end=(15, 20, 6))),
                     c.generate(2))
    self.assertEqual(((1, 1, 0), Box(start=(5, 7, 0), end=(20, 20, 6))),
                     c.generate(3))
    self.assertEqual(((0, 0, 1), Box(start=(0, 0, 4), end=(15, 13, 10))),
                     c.generate(4))
    self.assertEqual(((1, 0, 1), Box(start=(5, 0, 4), end=(20, 13, 10))),
                     c.generate(5))
    self.assertEqual(((0, 1, 1), Box(start=(0, 7, 4), end=(15, 20, 10))),
                     c.generate(6))
    self.assertEqual(((1, 1, 1), Box(start=(5, 7, 4), end=(20, 20, 10))),
                     c.generate(7))

    with self.assertRaises(ValueError):
      c.generate(8)

    # All calculated boxes have the full size
    for x in range(0, 8):
      b = c.generate(x)
      np.testing.assert_array_equal(c.box_size, b[1].size)

  def test_box_coordinate_to_index(self):
    c = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(20, 20, 10)),
        box_size=(15, 13, 6),
        box_overlap=(1, 2, 2))
    self.assertEqual(0, c.box_coordinate_to_index((0, 0, 0)))
    self.assertEqual(1, c.box_coordinate_to_index((1, 0, 0)))
    self.assertEqual(2, c.box_coordinate_to_index((0, 1, 0)))
    self.assertEqual(3, c.box_coordinate_to_index((1, 1, 0)))
    self.assertEqual(4, c.box_coordinate_to_index((0, 0, 1)))
    self.assertEqual(5, c.box_coordinate_to_index((1, 0, 1)))
    self.assertEqual(6, c.box_coordinate_to_index((0, 1, 1)))
    self.assertEqual(7, c.box_coordinate_to_index((1, 1, 1)))
    with self.assertRaises(ValueError):
      c.box_coordinate_to_index((1, 1, 2))

  def test_offset_to_index(self):
    outer_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(20, 20, 10))
    sub_box_size = (15, 13, 6)
    overlap = (1, 2, 2)
    c = m.BoxGenerator(outer_box, sub_box_size, overlap)

    with self.assertRaises(ValueError):
      c.offset_to_index(0, [-1])
    with self.assertRaises(ValueError):
      c.offset_to_index(0, [])
    with self.assertRaises(ValueError):
      c.offset_to_index(0, [1, 2, 3, 4, 5])

    # Off the front end.
    self.assertIsNone(c.offset_to_index(0, [-1, 0, 0]))
    self.assertIsNone(c.offset_to_index(0, [0, -1, 0]))
    self.assertIsNone(c.offset_to_index(0, [0, 0, -1]))

    # Okay.
    self.assertEqual(0, c.offset_to_index(0, [0, 0, 0]))
    self.assertEqual(1, c.offset_to_index(0, [1, 0, 0]))
    self.assertEqual(2, c.offset_to_index(0, [0, 1, 0]))
    self.assertEqual(4, c.offset_to_index(0, [0, 0, 1]))

    # Off the back end.
    self.assertIsNone(c.offset_to_index(0, [2, 0, 0]))

  def test_spatial_point_to_box_coordinates(self):

    def _maps_to(calc, coords, point):
      self.assertCountEqual(coords,
                            calc.spatial_point_to_box_coordinates(point))

    c = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(40, 40, 40)),
        box_size=(20, 20, 20),
        box_overlap=(10, 10, 10))

    _maps_to(c, [(0, 0, 0)], (0, 0, 0))
    _maps_to(c, [(0, 0, 0)], (9, 0, 0))
    _maps_to(c, [
        (0, 0, 0),
        (1, 0, 0),
    ], (10, 0, 0))
    _maps_to(c, [
        (0, 2, 0),
        (0, 2, 1),
        (1, 2, 0),
        (1, 2, 1),
    ], (10, 30, 10))
    _maps_to(c, [], (10, 40, 10))

    # Non-50% overlap
    c = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(40, 40, 40)),
        box_size=(20, 20, 20),
        box_overlap=(1, 1, 1))

    _maps_to(c, [(0, 0, 0)], (0, 0, 0))
    _maps_to(c, [(0, 0, 0)], (18, 0, 0))
    _maps_to(c, [
        (0, 0, 0),
        (1, 0, 0),
    ], (19, 0, 0))

    # Test non-zero offset
    c = m.BoxGenerator(
        Box(start=(50, 60, 70), size=(40, 40, 40)),
        box_size=(20, 20, 20),
        box_overlap=(3, 4, 5))
    _maps_to(c, [], (0, 0, 0))
    _maps_to(c, [(0, 0, 0)], (50, 60, 70))
    _maps_to(c, [
        (0, 0, 0),
        (1, 0, 0),
    ], (67, 60, 70))
    _maps_to(c, [
        (0, 0, 1),
        (1, 0, 1),
    ], (67, 60, 97))

    # Test back-shift; should have no effect
    c = m.BoxGenerator(
        Box(start=(50, 60, 70), size=(40, 40, 40)),
        box_size=(20, 20, 20),
        box_overlap=(3, 4, 5),
        back_shift_small_boxes=True)
    _maps_to(c, [], (0, 0, 0))
    _maps_to(c, [(0, 0, 0)], (50, 60, 70))
    _maps_to(c, [
        (0, 0, 0),
        (1, 0, 0),
    ], (67, 60, 70))
    _maps_to(c, [
        (0, 0, 1),
        (1, 0, 1),
    ], (67, 60, 97))

  def test_flow_test_uses(self):
    c = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(120, 120, 2)),
        box_size=(80, 80, 2),
        box_overlap=(40, 40))

    self.assertEqual(Box(start=(0, 0, 0), size=(120, 120, 2)), c.outer_box)
    self.assertEqual(Box(start=(0, 0, 0), size=(2, 2, 1)), c.output)
    self.assertEqual(c.num_boxes, 4)

    d = m.BoxGenerator(
        Box(start=(0, 0, 0, 0), size=(2, 2, 2, 4)), box_size=(1, 1, 2, 4))
    self.assertEqual(Box(start=(0, 0, 0, 0), size=(2, 2, 2, 4)), d.outer_box)
    self.assertEqual(Box(start=(0, 0, 0, 0), size=(2, 2, 1, 1)), d.output)
    self.assertEqual(d.num_boxes, 4)

  def test_batch(self):
    generator = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(20, 20, 10)),
        box_size=(15, 13, 6),
        box_overlap=(1, 2, 2))
    self.assertEqual(generator.num_boxes, 8)
    self.assertEqual([[
        Box(start=(0, 0, 0), size=(15, 13, 6)),
        Box(start=(14, 0, 0), size=(6, 13, 6)),
        Box(start=(0, 11, 0), size=(15, 9, 6))
    ],
                      [
                          Box(start=(14, 11, 0), size=(6, 9, 6)),
                          Box(start=(0, 0, 4), size=(15, 13, 6)),
                          Box(start=(14, 0, 4), size=(6, 13, 6))
                      ],
                      [
                          Box(start=(0, 11, 4), size=(15, 9, 6)),
                          Box(start=(14, 11, 4), size=(6, 9, 6))
                      ]], [list(b) for b in generator.batch(batch_size=3)])

    # Custom endpoint
    self.assertEqual([[
        Box(start=(0, 0, 0), size=(15, 13, 6)),
        Box(start=(14, 0, 0), size=(6, 13, 6)),
        Box(start=(0, 11, 0), size=(15, 9, 6))
    ], [
        Box(start=(14, 11, 0), size=(6, 9, 6)),
    ]], [list(b) for b in generator.batch(batch_size=3, end_index=4)])

  def test_tag_border_locations(self):
    generator = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(100, 100, 100)),
        box_size=(10, 10, 10),
        box_overlap=(0, 0, 0))
    total_edge_boxes = 0
    for i in range(generator.num_boxes):
      locs = generator.tag_border_locations(i)
      if np.any(locs[0]) or np.any(locs[1]):
        total_edge_boxes += 1
    self.assertEqual(488, total_edge_boxes)

  def test_overlapping_subboxes(self):
    calc = m.BoxGenerator(
        Box(start=(0, 0, 0), size=(2048, 2048, 1024)),
        box_size=(256, 256, 256),
        box_overlap=(128, 128, 128))

    overlapping = list(
        calc.overlapping_subboxes(Box(start=(123, 384, 789), size=(1, 1, 1))))
    size = (256, 256, 256)
    self.assertListEqual(overlapping, [
        Box(start=(0, 256, 640), size=size),
        Box(start=(0, 384, 640), size=size),
        Box(start=(0, 256, 768), size=size),
        Box(start=(0, 384, 768), size=size)
    ])


class MultiBoxGeneratorTest(absltest.TestCase):

  def test_multi_generator(self):
    outer_box = Box(start=(0, 0, 0), size=(20, 20, 10))
    sub_box_size = (15, 13, 6)
    overlap = (1, 2, 2)
    multi_generator = m.MultiBoxGenerator([outer_box, outer_box], sub_box_size,
                                          overlap)
    self.assertEqual(multi_generator.num_boxes, 16)

    self.assertEqual(multi_generator.index_to_generator_index(0), (0, 0))
    self.assertEqual(multi_generator.index_to_generator_index(7), (0, 7))
    self.assertEqual(multi_generator.index_to_generator_index(8), (1, 0))
    self.assertEqual(multi_generator.index_to_generator_index(14), (1, 6))
    self.assertRaises(ValueError, multi_generator.index_to_generator_index, 16)

    self.assertEqual([multi_generator.generate(i)[1] for i in range(8)],
                     list(multi_generator.generators[0].boxes))
    self.assertEqual([multi_generator.generate(i)[1] for i in range(8, 16)],
                     list(multi_generator.generators[1].boxes))


if __name__ == '__main__':
  absltest.main()

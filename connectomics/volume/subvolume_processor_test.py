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
"""Tests for subvolume_processor."""

from typing import Any

from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.volume import subvolume_processor
import numpy as np

BBox = bounding_box.BoundingBox
Processor = subvolume_processor.SubvolumeProcessor
_: Any = None


class SubvolumeProcessorTest(absltest.TestCase):

  def test_output_type(self):
    p = Processor()
    self.assertEqual(np.uint8, p.output_type(np.uint8))  # pytype: disable=wrong-arg-types  # numpy-scalars
    self.assertEqual(np.uint64, p.output_type(np.uint64))  # pytype: disable=wrong-arg-types  # numpy-scalars
    self.assertEqual(np.float32, p.output_type(np.float32))  # pytype: disable=wrong-arg-types  # numpy-scalars

  def test_output_num(self):
    p = Processor()
    self.assertEqual(subvolume_processor.OutputNums.SINGLE, p.output_num)

  def test_name_parts(self):
    p = Processor()
    self.assertEqual(('SubvolumeProcessor',), p.name_parts)

  def test_pixelsize(self):
    p = Processor()
    self.assertSameElements([8, 9, 33], p.pixelsize([8, 9, 33]))

  def test_num_channels(self):
    p = Processor()
    self.assertEqual(77, p.num_channels(77))

  def test_process(self):
    p = Processor()
    with self.assertRaises(NotImplementedError):
      p.process(_)

  def test_subvolume_size(self):
    p = Processor()
    self.assertIsNone(p.subvolume_size())

  def test_context(self):
    p = Processor()
    self.assertEqual(((0, 0, 0), (0, 0, 0)), p.context())

  def test_overlap(self):
    p = Processor()
    self.assertEqual((0, 0, 0), p.overlap())

  def test_set_effective_subvol_and_overlap(self):
    p = Processor()
    self.assertEqual(((0, 0, 0), (0, 0, 0)), p.context())
    p.set_effective_subvol_and_overlap([10, 11, 12], [2, 2, 2])
    self.assertEqual(((0, 0, 0), (0, 0, 0)), p.context())

  def test_crop_box(self):
    p = Processor()
    box = BBox([10, 11, 12], [4, 4, 4])

    # Assert that if we don't set the effective subvol and overlap, the base
    # class errors with a AttributeError due to _context not being initialized.
    with self.assertRaises(AttributeError):
      p.crop_box(box)

    # No overlap case, same size
    p.set_effective_subvol_and_overlap(box.size, [0, 0, 0])
    self.assertEqual(box, p.crop_box(box))

    # Overlap case, same size
    p.set_effective_subvol_and_overlap(box.size, [2, 2, 2])
    self.assertEqual(BBox([11, 12, 13], [2, 2, 2]), p.crop_box(box))

    # Non-even overlap case, same size
    p.set_effective_subvol_and_overlap(box.size, [3, 3, 3])
    self.assertEqual(BBox([11, 12, 13], [1, 1, 1]), p.crop_box(box))

    # No overlap case, different size
    p.set_effective_subvol_and_overlap([30, 30, 30], [0, 0, 0])
    self.assertEqual(box, p.crop_box(box))

    # # Overlap case, different size
    p.set_effective_subvol_and_overlap([30, 30, 30], [2, 2, 2])
    self.assertEqual(BBox([11, 12, 13], [2, 2, 2]), p.crop_box(box))

  def test_crop_box_and_data(self):
    p = Processor()
    box = BBox([10, 11, 12], [10, 10, 10])
    data = np.ones([10] * 4).cumsum(axis=0)

    # Assert that if we don't set the effective subvol and overlap, the base
    # class errors with a AttributeError due to _context not being initialized.
    with self.assertRaises(AttributeError):
      p.crop_box_and_data(box, data)

    p.set_effective_subvol_and_overlap(box.size, [0, 0, 0])
    subvol = p.crop_box_and_data(box, data)
    self.assertEqual(box, subvol.bbox)
    self.assertSequenceEqual([10, 10, 10, 10], subvol.data.shape)
    self.assertTrue(np.all(data == subvol.data))

    p.set_effective_subvol_and_overlap(box.size, [4, 4, 4])
    subvol = p.crop_box_and_data(box, data)
    self.assertEqual(BBox([12, 13, 14], [6, 6, 6]), subvol.bbox)
    self.assertSequenceEqual([10, 6, 6, 6], subvol.data.shape)
    self.assertTrue(np.all(data[:, 2:8, 2:8, 2:8] == subvol.data))


if __name__ == '__main__':
  absltest.main()

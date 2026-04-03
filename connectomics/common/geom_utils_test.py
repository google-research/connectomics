# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.common import geom_utils
import numpy as np


class DownsampleAreaTest(absltest.TestCase):

  def test_uniform_data_exact(self):
    """A constant volume should downsample to the same constant."""
    shape = (6, 8, 10)
    val = 42
    data = np.full(shape, val, dtype=np.int64)
    scale = np.array([2, 2, 2])
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=shape[::-1])

    svt = geom_utils.integral_image(data)
    _, out = geom_utils.downsample_area(svt, box, scale, np.dtype(np.uint8))
    np.testing.assert_array_equal(out, val)

  def test_uniform_data_with_remainder(self):
    """A constant volume with remainders should still downsample to the same constant."""
    shape = (5, 7, 9)
    val = 100
    data = np.full(shape, val, dtype=np.int64)
    scale = np.array([2, 2, 2])
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=shape[::-1])

    svt = geom_utils.integral_image(data)
    _, out = geom_utils.downsample_area(svt, box, scale, np.dtype(np.uint8))
    np.testing.assert_array_equal(out, val)

  def test_output_box_coordinates(self):
    """Output box start is correctly aligned to the downsampled grid."""
    shape = (5, 7, 9)
    data = np.ones(shape, dtype=np.int64)
    scale = np.array([2, 2, 2])
    box = bounding_box.BoundingBox(start=(10, 20, 30), size=shape[::-1])

    svt = geom_utils.integral_image(data)
    out_box, out = geom_utils.downsample_area(
        svt, box, scale, np.dtype(np.uint8)
    )

    # The output box start should be in downsampled coordinates:
    # (box.start + off) // scale
    np.testing.assert_array_equal(out_box.start, [5, 10, 15])
    np.testing.assert_array_equal(out_box.size, np.array(out.shape[1:][::-1]))


class IntegralImageTest(absltest.TestCase):

  def test_2d(self):
    data = np.array([[1, 2], [3, 4]])
    ii = geom_utils.integral_image(data)
    # Top-left origin should be padded with zeros.
    self.assertEqual(ii[0, 0], 0)
    self.assertEqual(ii[0, 1], 0)
    self.assertEqual(ii[1, 0], 0)
    # Full sum.
    self.assertEqual(ii[2, 2], 10)

  def test_3d(self):
    data = np.ones((2, 3, 4), dtype=int)
    ii = geom_utils.integral_image(data)
    self.assertEqual(ii[-1, -1, -1], 24)


class QueryIntegralImageTest(absltest.TestCase):

  def test_2d_uniform(self):
    data = np.ones((4, 6), dtype=int)
    ii = geom_utils.integral_image(data)
    result = geom_utils.query_integral_image(ii, diam=[2, 3])
    # Each 2x3 block sums to 6.
    np.testing.assert_array_equal(result, 6)

  def test_3d_with_stride(self):
    data = np.ones((4, 6, 8), dtype=int)
    ii = geom_utils.integral_image(data)
    result = geom_utils.query_integral_image(
        ii, diam=[2, 2, 2], stride=[2, 2, 2]
    )
    np.testing.assert_array_equal(result, 8)


if __name__ == '__main__':
  absltest.main()

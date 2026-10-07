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
"""Area downsampling must preserve fractional values for floating outputs."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized
from connectomics.common import bounding_box, geom_utils
import numpy as np


def direct_blocks(data, start, scale, mask=None):
  start = np.asarray(start)
  offset = (-start) % scale
  data = data[tuple(slice(int(o), None) for o in offset[::-1])]
  if mask is not None:
    mask = mask[tuple(slice(int(o), None) for o in offset[::-1])]
  steps = scale[::-1]
  shape = (np.array(data.shape) + steps - 1) // steps
  out = np.empty(tuple(shape), dtype=np.float64)
  for index in itertools.product(*(range(int(n)) for n in shape)):
    slices = tuple(
        slice(i * int(s), (i + 1) * int(s)) for i, s in zip(index, steps)
    )
    block = data[slices]
    if mask is not None:
      block = block[~mask[slices]]
    out[index] = np.mean(block, dtype=np.float64) if block.size else np.nan
  return (start + offset) // scale, out[None, ...]


class FloatDownsampleTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[np.float32, np.float64], value=[0.25, -0.375, 1.75]
  )
  def test_constant_fractional_volumes_remain_constant(self, dtype, value):
    data = np.full((3, 5, 7), value, dtype=dtype)
    scale = np.array([2, 3, 2])
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=data.shape[::-1])
    _, actual = geom_utils.downsample_area(
        geom_utils.integral_image(data), box, scale, dtype
    )
    np.testing.assert_allclose(actual, value, rtol=0, atol=1e-7)
    self.assertEqual(actual.dtype, dtype)

  @parameterized.parameters((0, 0, 0), (1, 2, 3), (-1, 1, -3))
  def test_aligned_and_partial_blocks_match_direct_means(self, x, y, z):
    data = (np.arange(3 * 5 * 7).reshape(3, 5, 7) % 13) / 16 - 0.25
    scale = np.array([2, 3, 2])
    box = bounding_box.BoundingBox(start=(x, y, z), size=data.shape[::-1])
    svt = geom_utils.integral_image(data)
    before = svt.copy()
    out_box, actual = geom_utils.downsample_area(
        svt, box, scale, np.dtype('float64')
    )
    expected_start, expected = direct_blocks(data, (x, y, z), scale)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(out_box.start, expected_start)
    np.testing.assert_array_equal(out_box.size, actual.shape[:0:-1])
    np.testing.assert_array_equal(svt, before)

  def test_masked_blocks_average_only_valid_fractional_values(self):
    data = np.arange(4 * 4 * 4).reshape(4, 4, 4) / 64
    mask = np.zeros(data.shape, dtype=bool)
    mask[:2, :2, :2] = True
    mask[2, 2, 2] = True
    masked = np.where(mask, 0, data)
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=data.shape[::-1])
    scale = np.array([2, 2, 2])
    _, actual = geom_utils.downsample_area(
        geom_utils.integral_image(masked),
        box,
        scale,
        np.float32,
        mask_svt=geom_utils.integral_image(mask),
    )
    _, expected = direct_blocks(data, (0, 0, 0), scale, mask)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, equal_nan=True)
    self.assertTrue(np.isnan(actual[0, 0, 0, 0]))

  def test_integer_inputs_can_have_fractional_floating_outputs(self):
    data = np.array([[[0, 1], [1, 1]], [[0, 1], [0, 0]]], dtype=np.uint8)
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=(2, 2, 2))
    _, actual = geom_utils.downsample_area(
        geom_utils.integral_image(data), box, np.array([2, 2, 2]), np.float64
    )
    np.testing.assert_array_equal(actual, 0.5)

  @parameterized.parameters(np.uint8, np.int16, np.bool_)
  def test_nonfloating_outputs_retain_round_then_cast_behavior(self, dtype):
    data = np.array([[[0.0, 1.0, 1.0, 2.0, 2.0, 3.0]]])
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=(6, 1, 1))
    _, actual = geom_utils.downsample_area(
        geom_utils.integral_image(data), box, np.array([2, 1, 1]), dtype
    )
    expected = np.round([0.5, 1.5, 2.5]).astype(dtype).reshape(1, 1, 1, 3)
    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()

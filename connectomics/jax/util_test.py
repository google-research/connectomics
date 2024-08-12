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
"""Tests for util."""

from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.jax import util
import jax
import jax.numpy as jnp


class UtilTest(absltest.TestCase):

  def test_center_crop_in_all_dimensions(self):
    original = jnp.arange(0, 1000, 1, dtype=int).reshape((10, 10, 10))

    # Crop to a larger size in one of dimensions
    cropped = util.center_crop_in_all_dimensions(original, (2, 3000, 2))
    self.assertEqual(cropped.shape, (2, 10, 2))

    # Really cropping in all dimensions
    cropped = util.center_crop_in_all_dimensions(original, (2, 2, 2))
    self.assertEqual(cropped.shape, (2, 2, 2))
    self.assertEqual((cropped == jnp.asarray([[[444, 445], [454, 455]],
                                              [[544, 545], [554, 555]]])).all(),
                     True)

    # No crop in any dimension.
    cropped = util.center_crop_in_all_dimensions(cropped, (20, 20, 20))
    self.assertEqual(cropped.shape, (2, 2, 2))
    self.assertEqual((cropped == jnp.asarray([[[444, 445], [454, 455]],
                                              [[544, 545], [554, 555]]])).all(),
                     True)

    # Crop requested, no symmetric crop possible. We expect less crop on the
    # right
    original = jnp.arange(0, 125, 1, dtype=int).reshape((5, 5, 5))
    cropped = util.center_crop_in_all_dimensions(original, (2, 2, 2))

    self.assertEqual(cropped.shape, (2, 2, 2))
    self.assertEqual((cropped == jnp.asarray([[[31, 32], [36, 37]],
                                              [[56, 57], [61, 62]]])).all(),
                     True)

  def test_center_crop(self):
    original = jnp.arange(0, 1000, 1, dtype=int).reshape((10, 10, 10))

    @jax.jit
    def _crop(x):
      return util.center_crop(x, (2,))

    # Crop to a larger size in one of dimensions
    cropped = _crop(original)
    self.assertEqual(cropped.shape, (10, 2, 10))

  def test_pad_symmetrically_in_all_dimensions(self):
    original = jnp.arange(0, 8, 1, dtype=int).reshape((2, 2, 2))

    # Need to pad asymmetrically
    padded = util.pad_symmetrically_in_all_dimensions(original, (3, 3, 3))
    self.assertEqual(padded.shape, (3, 3, 3))
    self.assertEqual((padded == jnp.asarray([[[0, 1, 0], [2, 3, 0], [0, 0, 0]],
                                             [[4, 5, 0], [6, 7, 0], [0, 0, 0]],
                                             [[0, 0, 0], [0, 0, 0],
                                              [0, 0, 0]]])).all(), True)

    # Regular symmetric padding
    padded = util.pad_symmetrically_in_all_dimensions(original, (4, 4, 4))
    self.assertEqual(padded.shape, (4, 4, 4))
    self.assertEqual((padded == jnp.asarray([[[0, 0, 0, 0], [0, 0, 0, 0],
                                              [0, 0, 0, 0], [0, 0, 0, 0]],
                                             [[0, 0, 0, 0], [0, 0, 1, 0],
                                              [0, 2, 3, 0], [0, 0, 0, 0]],
                                             [[0, 0, 0, 0], [0, 4, 5, 0],
                                              [0, 6, 7, 0], [0, 0, 0, 0]],
                                             [[0, 0, 0, 0], [0, 0, 0, 0],
                                              [0, 0, 0, 0], [0, 0, 0,
                                                             0]]])).all(), True)

  def test_crop_bounding_box(self):
    original = bounding_box.BoundingBox(start=(10, 10, 10), end=(20, 20, 20))
    cropped = util.center_crop_bounding_box(original, (2, 2, 2))
    self.assertEqual(tuple(cropped.start), (14, 14, 14))
    self.assertEqual(tuple(cropped.size), (2, 2, 2))

    cropped = util.center_crop_bounding_box(original, (3, 3, 3))
    self.assertEqual(tuple(cropped.start), (13, 13, 13))
    self.assertEqual(tuple(cropped.size), (3, 3, 3))


if __name__ == '__main__':
  absltest.main()

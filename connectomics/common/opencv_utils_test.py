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
"""Tests for opencv_utils."""

from absl.testing import absltest
from connectomics.common import opencv_utils
import cv2 as cv
import numpy as np
import skimage
import skimage.transform


class OpencvUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_img = skimage.data.immunohistochemistry()[:50, :75, 0].T

  def test_warp_identity_transform(self):
    transform = np.array(
        [[1., 0., 0.],
         [0., 1., 0.]], dtype=np.float32)
    res = opencv_utils.warp_affine(self._test_img, transform)
    np.testing.assert_allclose(self._test_img, res, rtol=1e-5)

  def test_affine_alignment(self,
                            translate_x: float = 1.,
                            translate_y: float = 5.,
                            scale_x: float = 1.1,
                            scale_y: float = 0.9,
                            rotate: float = 0.02,
                            shear: float = 0.01):
    fix = np.pad(self._test_img / 255., 10).astype(np.float32)
    transform = skimage.transform.AffineTransform(
        scale=(scale_x, scale_y),
        translation=(translate_x, translate_y),
        rotation=rotate,
        shear=shear)
    transform_expected = cv.invertAffineTransform(transform.params[:2, ...])
    mov = skimage.transform.warp(fix, inverse_map=transform.inverse)
    _, transform_optim = opencv_utils.optim_transform(fix, mov)
    res = opencv_utils.warp_affine(mov, transform_optim)
    assert res.shape == fix.shape
    np.testing.assert_allclose(transform_expected, transform_optim, atol=0.2)


if __name__ == '__main__':
  absltest.main()

# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Tests for ellipticity metric."""

from absl.testing import absltest
from connectomics.metrics.ellipticity import _ellipticity_factor
from connectomics.metrics.ellipticity import compute_ellipticity
import numpy as np
from skimage import transform
import skimage.morphology as morph


class EllipticityTest(absltest.TestCase):

  _test_imgs = {
      '2d_circle_small': morph.disk(10),
      '2d_circle_medium': morph.disk(100),
      '2d_circle_large': morph.disk(1000),
      '2d_ellipse_small': np.array(
          transform.rescale(morph.disk(10), scale=(1.0, 2.0)), dtype=bool),
      '2d_ellipse_medium': np.array(
          transform.rescale(morph.disk(100), scale=(1.0, 2.0)), dtype=bool),
      '2d_ellipse_large': np.array(
          transform.rescale(morph.disk(1000), scale=(1.0, 2.0)), dtype=bool),
      '2d_square': morph.square(25),
      '2d_rectangle': morph.rectangle(15, 10),
      '2d_diamond': morph.diamond(7),
      '2d_octagon': morph.octagon(50, 10),
      '2d_star': morph.star(75),
      '3d_ball': morph.ball(50),
      '3d_ellipsoid': np.array(
          transform.rescale(morph.ball(50), scale=(1.0, 1.5, 2.0)), dtype=bool),
      '3d_cube': morph.cube(11),
      '3d_octahedron': morph.octahedron(5),
    }

  def test_factor(self):
    assert _ellipticity_factor(2) == 1. / (4 * np.pi)
    assert _ellipticity_factor(3) == 1. / (20. * ((np.sqrt(5) * np.pi) / 3.))
    assert _ellipticity_factor(3) == (3. * np.sqrt(5)) / (100. * np.pi)

  def test_2d_positive_examples_circles(self):
    for name, precision in zip(
        ['2d_circle_small', '2d_circle_medium', '2d_circle_large'],
        [2, 3, 4]):
      np.testing.assert_almost_equal(compute_ellipticity(self._test_imgs[name]),
                                     desired=1.0, decimal=precision)

  def test_2d_positive_examples_ellipses(self):
    for name, precision in zip(
        ['2d_ellipse_small', '2d_ellipse_medium', '2d_ellipse_large'],
        [2, 3, 4]):
      np.testing.assert_almost_equal(compute_ellipticity(self._test_imgs[name]),
                                     desired=1.0, decimal=precision)

  def test_2d_negative_examples(self):
    for name in ['2d_rectangle', '2d_square', '2d_star', '2d_octagon']:
      np.testing.assert_array_less(
          compute_ellipticity(self._test_imgs[name]), 0.98)

  def test_3d_positive_examples(self):
    for name in ['3d_ball', '3d_ellipsoid']:
      np.testing.assert_almost_equal(compute_ellipticity(self._test_imgs[name]),
                                     desired=1.0, decimal=3)

  def test_3d_negative_examples(self):
    for name in ['3d_cube', '3d_octahedron']:
      np.testing.assert_array_less(
          compute_ellipticity(self._test_imgs[name]), 0.9)


if __name__ == '__main__':
  absltest.main()

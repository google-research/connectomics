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
"""Tests for base."""

from absl.testing import absltest
from connectomics.common import array
from connectomics.volume import base
import numpy as np
import numpy.testing as npt


class ShimVolume(base.BaseVolume):

  def __init__(self, *args, **kwargs):
    super(*args, **kwargs)
    self.called = False

  @property
  def shape(self) -> array.Tuple4i:
    return (1, 12, 11, 10)


class BaseVolumeTest(absltest.TestCase):

  def test_not_implemented(self):
    v = base.BaseVolume()

    for field in [
        'volume_size', 'voxel_size', 'shape', 'ndim', 'dtype', 'bounding_boxes'
    ]:
      with self.assertRaises(NotImplementedError):
        _ = getattr(v, field)

  def test_get_points(self):
    tself = self

    class GetPointsVolume(ShimVolume):

      def get_points(self, points):
        self.called = True
        tself.assertLen(points, 4)
        for i in range(1, 4):
          tself.assertLen(points[i], 3)
        return np.random.uniform(size=[1, 3, 3, 3])

    v = GetPointsVolume()
    _ = v[0, (1, 2, 3), (4, 5, 6), (7, 8, 9)]
    self.assertTrue(v.called)

    v = GetPointsVolume()
    _ = v[0, [1, 2, 3], [4, 5, 6], [7, 8, 9]]
    self.assertTrue(v.called)

  def test_get_slices(self):
    tself = self

    expected = (
        slice(0, 1, None),
        slice(1, 3, 1),
        slice(5, 11, 1),
        slice(0, 10, 1),
    )

    class GetSlicesVolume(ShimVolume):

      def get_slices(self, slices):
        self.called = True
        tself.assertLen(slices, 4)
        tself.assertEqual(expected, slices)
        shape = [(i.stop - i.start) for i in slices]
        data = np.random.uniform(size=shape)
        return data

    v = GetSlicesVolume()
    sv = v[0, 1:3, 5:, :]
    self.assertTrue(v.called)
    self.assertEqual(sv.data.shape, (1, 2, 6, 10))
    self.assertEqual(tuple(sv.bbox.start), (0, 5, 1))
    self.assertEqual(tuple(sv.bbox.end), (10, 11, 3))
    self.assertEqual(tuple(sv.bbox.size), (10, 6, 2))

    # Test accessing the data directly without the subvolume abstraction.
    v.called = False
    data = v.asarray[0, 1:3, 5:, :]
    self.assertTrue(v.called)
    self.assertEqual(data.shape, (1, 2, 6, 10))

  def test_write_points(self):
    tself = self

    expected = [1, 2, 3]

    class WritePointsVolume(ShimVolume):

      def write_points(self, points, values):
        self.called = True
        tself.assertLen(points, 4)
        for i in range(1, 4):
          tself.assertLen(points[i], 3)
        tself.assertLen(values, 3)
        npt.assert_array_equal(values, expected)

    v = WritePointsVolume()
    v[0, (1, 2, 3), (4, 5, 6), (7, 8, 9)] = np.array(expected)
    self.assertTrue(v.called)

  def test_write_slices(self):
    tself = self

    expected_slices = (
        slice(0, 1, None),
        slice(1, 3, 1),
        slice(5, 11, 1),
        slice(0, 10, 1),
    )

    expected_data = np.random.uniform(size=[1, 12, 11, 10])

    class WriteSlicesVolume(ShimVolume):

      def write_slices(self, slices, data):
        self.called = True
        tself.assertLen(slices, 4)
        tself.assertEqual(expected_slices, slices)
        npt.assert_array_equal(expected_data, data)

    v = WriteSlicesVolume()
    v[0, 1:3, 5:, :] = expected_data
    self.assertTrue(v.called)


if __name__ == '__main__':
  absltest.main()

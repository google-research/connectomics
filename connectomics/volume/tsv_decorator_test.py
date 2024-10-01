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
"""Tests for tsv_decorator."""

import typing
from typing import Any

from absl.testing import absltest
from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.common import tuples
from connectomics.volume import base as base_volume
from connectomics.volume import descriptor as vd
from connectomics.volume import metadata
from connectomics.volume import tsv_decorator
import numpy as np
import numpy.testing as npt
import numpy.typing as nptyping

BBox = bounding_box.BoundingBox

_: Any = None


# TODO(timblakely): Create an common in-memory volume implementation. Would be
# useful in both tests and in temporary volume situations.
class DummyVolume(base_volume.Volume):

  def __init__(
      self,
      volume_size: tuple[int, int, int],
      voxel_size: tuple[int, int, int],
      bounding_boxes: list[BBox],
      data: np.ndarray,
      dtype: nptyping.DTypeLike,
  ):
    super().__init__(
        metadata.VolumeMetadata(
            path='none',
            volume_size=tuples.XYZ(*volume_size),
            pixel_size=tuples.XYZ(*voxel_size),
            bounding_boxes=bounding_boxes,
            dtype=dtype,
        )
    )
    self._data = data

  def __getitem__(self, ind):
    return self._data[ind]

  def get_points(self, points: array.PointLookups) -> np.ndarray:
    num_points = len(points[1])
    coordinates = np.empty([num_points, 3], dtype=np.int64)
    coordinates[:, 0] = points[1]
    coordinates[:, 1] = points[2]
    coordinates[:, 2] = points[3]
    return self._data[points[0], tuple(np.array(coordinates).T.tolist())]

  def get_slices(self, slices: array.CanonicalSlice) -> np.ndarray:
    return self._data[slices]


def _make_dummy_vol() -> tuple[DummyVolume, BBox, np.ndarray]:
  bbox = BBox([100, 200, 300], [20, 50, 100])
  data = np.zeros(bbox.size)
  data[0] = 1
  data = np.cumsum(
      np.cumsum(np.cumsum(data, axis=0), axis=1), axis=2, dtype=np.uint64)
  data = data[np.newaxis]
  vol = DummyVolume(
      (3000, 2000, 1000), (8, 8, 33), [bbox], data, dtype=np.uint64
  )
  return vol, bbox, data


class DecoratorTest(absltest.TestCase):

  def test_dummy_volume(self):
    vol, bbox, data = _make_dummy_vol()
    self.assertEqual((3000, 2000, 1000), vol.volume_size)
    self.assertLen(vol.bounding_boxes, 1)
    self.assertEqual([bbox], vol.bounding_boxes)
    self.assertEqual((8, 8, 33), vol.pixel_size)
    self.assertEqual(np.uint64, vol.dtype)
    self.assertEqual(4, vol.ndim)
    self.assertEqual((1, 1000, 2000, 3000), vol.shape)
    npt.assert_array_equal(data, vol._data)
    npt.assert_array_equal(data[0], vol[0])
    npt.assert_array_equal(data[..., 2:5], vol[..., 2:5])

  def test_upscale(self):
    vol, bbox, data = _make_dummy_vol()
    scale = np.array((2, 2, 1))
    upscaled = tsv_decorator.Upsample(vol, scale)
    self.assertEqual((6000, 4000, 1000), tuple(upscaled.volume_size))
    self.assertEqual((4., 4., 33.), tuple(upscaled.voxel_size))
    self.assertEqual((1, 1000, 4000, 6000), tuple(upscaled.shape))
    self.assertLen(upscaled.bounding_boxes, 1)
    self.assertEqual(
        BBox(bbox.start * (2, 2, 1), bbox.size * (2, 2, 1)),
        upscaled.bounding_boxes[0])

    expected = data[0, 1, 1, 5].ravel()[0]
    # TODO(timblakely): Figure out why pytype thinks this is an error
    self.assertEqual(upscaled[0, 1, 2, 10].data.ravel()[0], expected)  # pytype: disable=attribute-error

    self.assertTrue(np.all(upscaled[0, 0:2, 2:4, 10].data == expected))
    self.assertFalse(np.all(upscaled[0, 1:3, 3:5, 10].data == expected))


class CustomDecorator(tsv_decorator.VolumeDecorator):
  pass


class CustomDecoratorFactory(tsv_decorator.DecoratorFactory):
  called: bool = False

  def __init__(self, *args, **kwargs):
    self.called = False

  def make_decorator(self, wrapped_volume: base_volume.Volume, name: str,
                     *args: list[Any],
                     **kwargs: dict[str, Any]) -> tsv_decorator.VolumeDecorator:
    if name == 'CustomDecorator':
      self.called = True
      return CustomDecorator(wrapped_volume)
    return tsv_decorator.GlobalsDecoratorFactory().make_decorator(
        wrapped_volume, name, *args, **kwargs)


class DecoratorFactoryTest(absltest.TestCase):

  def test_default_loader(self):
    descriptor: vd.VolumeDescriptor = vd.VolumeDescriptor.from_json("""{
          "decorator_specs": [{
            "decorator": "Upsample",
            "args": [2,2,1]
          }]
        }""")
    vol, _, _ = _make_dummy_vol()
    decorated = tsv_decorator.from_specs(vol, descriptor.decorator_specs)
    self.assertIsInstance(decorated, tsv_decorator.Upsample)

    # Ensure it loads only from `globals()`, aka tsv_decorator.py.
    descriptor: vd.VolumeDescriptor = vd.VolumeDescriptor.from_json("""{
          "decorator_specs": [{
            "decorator": "CustomDecorator"
          }]
        }""")

    with self.assertRaises(KeyError):
      decorated = tsv_decorator.from_specs(vol, descriptor.decorator_specs)

  def test_custom_loader(self):

    descriptor: vd.VolumeDescriptor = vd.VolumeDescriptor.from_json("""{
          "decorator_specs": [{
            "decorator": "CustomDecorator"
          }]
        }""")

    vol, _, _ = _make_dummy_vol()
    factory = CustomDecoratorFactory()

    decorated = tsv_decorator.from_specs(
        vol, descriptor.decorator_specs, decorator_factory=factory)
    self.assertIsInstance(decorated, CustomDecorator)
    self.assertIs(decorated._wrapped, vol)
    self.assertTrue(factory.called)

  def test_cascading_decorators(self):

    descriptor: vd.VolumeDescriptor = vd.VolumeDescriptor.from_json("""{
          "decorator_specs": [{
            "decorator": "Upsample",
            "args": [2,2,1]
          },
          {
            "decorator": "CustomDecorator"
          }]
        }""")

    vol, _, _ = _make_dummy_vol()
    factory = CustomDecoratorFactory()

    decorated = tsv_decorator.from_specs(
        vol, descriptor.decorator_specs, decorator_factory=factory)
    self.assertIsInstance(decorated, CustomDecorator)
    self.assertIsInstance(decorated._wrapped, tsv_decorator.Upsample)
    decorated = typing.cast(tsv_decorator.VolumeDecorator, decorated._wrapped)
    self.assertIsInstance(decorated._wrapped, DummyVolume)
    self.assertIs(decorated._wrapped, vol)
    self.assertTrue(factory.called)


if __name__ == '__main__':
  absltest.main()

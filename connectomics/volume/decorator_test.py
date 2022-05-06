"""Tests for subvolume_processor."""

import typing
from typing import Any, Sequence, Tuple

from absl.testing import absltest
from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.volume import base as base_volume
from connectomics.volume import decorator
import numpy as np
import numpy.testing as npt

BBox = bounding_box.BoundingBox

_: Any = None


# TODO(timblakely): Create an common in-memory volume implementation. Would be
# useful in both tests and in temporary volume situations.
class DummyVolume(base_volume.BaseVolume):

  def __init__(self, volume_size: Sequence[int], voxel_size: Sequence[int],
               bounding_boxes: list[BBox], data: np.ndarray):
    self._volume_size = tuple(volume_size)
    self._voxel_size = tuple(voxel_size)
    self._bounding_boxes = bounding_boxes
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

  @property
  def volume_size(self) -> array.Tuple3i:
    return self._volume_size

  @property
  def voxel_size(self) -> array.Tuple3i:
    return self._voxel_size

  @property
  def shape(self) -> array.Tuple4i:
    return (1,) + tuple(self._volume_size[::-1])

  @property
  def ndim(self) -> int:
    return len(self._data.shape)

  @property
  def dtype(self) -> np.dtype:
    return self._data.dtype

  @property
  def bounding_boxes(self) -> list[BBox]:
    return self._bounding_boxes


def _make_dummy_vol() -> Tuple[DummyVolume, BBox, np.ndarray]:
  bbox = BBox([100, 200, 300], [20, 50, 100])
  data = np.zeros(bbox.size)
  data[0] = 1
  data = np.cumsum(
      np.cumsum(np.cumsum(data, axis=0), axis=1), axis=2, dtype=np.uint64)
  data = data[np.newaxis]
  vol = DummyVolume([3000, 2000, 1000], (8, 8, 33), [bbox], data)
  return vol, bbox, data


class DecoratorTest(absltest.TestCase):

  def test_dummy_volume(self):
    vol, bbox, data = _make_dummy_vol()
    self.assertEqual((3000, 2000, 1000), vol.volume_size)
    self.assertLen(vol.bounding_boxes, 1)
    self.assertEqual([bbox], vol.bounding_boxes)
    self.assertEqual((8, 8, 33), vol.voxel_size)
    self.assertEqual(np.uint64, vol.dtype)
    self.assertEqual(4, vol.ndim)
    self.assertEqual((1, 1000, 2000, 3000), vol.shape)
    npt.assert_array_equal(data, vol._data)
    npt.assert_array_equal(data[0], vol[0])
    npt.assert_array_equal(data[..., 2:5], vol[..., 2:5])

  def test_upscale(self):
    vol, bbox, data = _make_dummy_vol()
    scale = np.array((2, 2, 1))
    upscaled = decorator.Upsample(vol, scale)
    self.assertEqual((6000, 4000, 1000), tuple(upscaled.volume_size))
    self.assertEqual((4., 4., 33.), tuple(upscaled.voxel_size))
    self.assertEqual((1, 1000, 4000, 6000), tuple(upscaled.shape))
    self.assertLen(upscaled.bounding_boxes, 1)
    self.assertEqual(
        BBox(bbox.start * (2, 2, 1), bbox.size * (2, 2, 1)),
        upscaled.bounding_boxes[0])

    expected = data[0, 1, 1, 5].ravel()[0]
    self.assertEqual(upscaled[0, 1, 2, 10].ravel()[0], expected)

    self.assertTrue(np.all(upscaled[0, 0:2, 2:4, 10] == expected))
    self.assertFalse(np.all(upscaled[0, 1:3, 3:5, 10] == expected))


class CustomDecorator(decorator.VolumeDecorator):
  pass


class CustomDecoratorFactory(decorator.DecoratorFactory):
  called: bool = False

  def __init__(self, *args, **kwargs):
    self.called = False

  def make_decorator(self, wrapped_volume: base_volume.BaseVolume, name: str,
                     *args: list[Any],
                     **kwargs: dict[str, Any]) -> decorator.VolumeDecorator:
    if name == 'CustomDecorator':
      self.called = True
      return CustomDecorator(wrapped_volume)
    return decorator.GlobalsDecoratorFactory().make_decorator(
        wrapped_volume, name, *args, **kwargs)


class DecoratorFactoryTest(absltest.TestCase):

  def test_default_loader(self):
    bv = base_volume
    descriptor: bv.VolumeDescriptor = bv.VolumeDescriptor.from_json("""{
          "decorator_specs": [{
            "decorator": "Upsample",
            "args": [2,2,1]
          }]
        }""")
    vol, _, _ = _make_dummy_vol()
    decorated = decorator.from_specs(vol, descriptor.decorator_specs)
    self.assertIsInstance(decorated, decorator.Upsample)

    # Ensure it loads only from `globals()`, aka decorator.py.
    descriptor: base_volume.VolumeDescriptor = bv.VolumeDescriptor.from_json(
        """{
          "decorator_specs": [{
            "decorator": "CustomDecorator"
          }]
        }""")

    with self.assertRaises(KeyError):
      decorated = decorator.from_specs(vol, descriptor.decorator_specs)

  def test_custom_loader(self):
    bv = base_volume

    descriptor: base_volume.VolumeDescriptor = bv.VolumeDescriptor.from_json(
        """{
          "decorator_specs": [{
            "decorator": "CustomDecorator"
          }]
        }""")

    vol, _, _ = _make_dummy_vol()
    factory = CustomDecoratorFactory()

    decorated = decorator.from_specs(
        vol, descriptor.decorator_specs, decorator_factory=factory)
    self.assertIsInstance(decorated, CustomDecorator)
    self.assertIs(decorated._wrapped, vol)
    self.assertTrue(factory.called)

  def test_cascading_decorators(self):
    bv = base_volume

    descriptor: base_volume.VolumeDescriptor = bv.VolumeDescriptor.from_json(
        """{
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

    decorated = decorator.from_specs(
        vol, descriptor.decorator_specs, decorator_factory=factory)
    self.assertIsInstance(decorated, CustomDecorator)
    self.assertIsInstance(decorated._wrapped, decorator.Upsample)
    decorated = typing.cast(decorator.VolumeDecorator, decorated._wrapped)
    self.assertIsInstance(decorated._wrapped, DummyVolume)
    self.assertIs(decorated._wrapped, vol)
    self.assertTrue(factory.called)


if __name__ == '__main__':
  absltest.main()

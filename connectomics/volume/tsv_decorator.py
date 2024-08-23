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
"""Volume decorators for modifying volumes on-the-fly."""

from typing import Any, Dict, Optional

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.volume import base
import numpy as np

# List of python dicts of specs to decorate a volume. Expected to have the
# following fields:
#   "decorator": Name of the decorator class. By default the decorator is looked
#     up via the GlobalsDecoratorFactory, which simply searches `globals()` for
#     a class of the same name.
#   "args": Optional. List of decorator-specific arguments.
#   "kwargs": Optional. Dict of decorator-specific keyword args.
DecoratorSpec = Dict[str, Any]


class DecoratorFactory:
  """Constructs a VolumeDecorator based on a name and arguments."""

  def make_decorator(self, wrapped_volume: base.Volume, name: str,
                     *args: list[Any],
                     **kwargs: dict[str, Any]) -> 'VolumeDecorator':
    raise NotImplementedError()


class GlobalsDecoratorFactory:
  """Loads VolumeDecorators from globals()."""

  def make_decorator(self, wrapped_volume: base.Volume, name: str,
                     *args: list[Any],
                     **kwargs: dict[str, Any]) -> 'VolumeDecorator':
    decorator_ctor = globals()[name]
    return decorator_ctor(wrapped_volume, *args, **kwargs)


def from_specs(volume: base.Volume,
               specs: list[DecoratorSpec],
               decorator_factory: Optional[DecoratorFactory] = None):
  """Decorates the given volume from the given specs.

  Args:
    volume: Base volume to wrap.
    specs: Sequence of decoration specs, described below.
    decorator_factory: Optional. Defaults to loading decorators from this file
      (`globals()`). If decorators need to be loaded from other modules,
      subclass DecoratorFactory and do path manipulation there.

  Returns:
    Decorated volume.  (Can return original volume if specs is empty.)

  A spec is a dict having fields:
    decorator: StrDecoratorClassName
    args: [sequence, of, decorator, args]
    kwargs: {dict_of: decorator_args}

  The spec sequence will be used to chain a sequence of decorators onto given
  volume, in left to right order.
  """

  if decorator_factory is None:
    decorator_factory = GlobalsDecoratorFactory()

  for s in specs:
    args = s.get('args', [])
    kwargs = s.get('kwargs', {})
    volume = decorator_factory.make_decorator(volume, s['decorator'], args,
                                              **kwargs)
  return volume


class VolumeDecorator(base.Volume):
  """Delegates to wrapped volumes, optionally applying transforms."""

  wrapped: base.Volume

  def __init__(self, wrapped: base.Volume):
    self._wrapped = wrapped

  def get_points(self, points: array.PointLookups) -> np.ndarray:
    return self._wrapped.get_points(points)

  def get_slices(self, slices: array.CanonicalSlice) -> np.ndarray:
    return self._wrapped.get_slices(slices)

  @property
  def volume_size(self) -> array.Tuple3i:
    return self._wrapped.volume_size

  @property
  def voxel_size(self) -> array.Tuple3f:
    return self._wrapped.pixel_size

  @property
  def shape(self) -> array.Tuple4i:
    return self._wrapped.shape

  @property
  def ndim(self) -> int:
    return self._wrapped.ndim

  @property
  def dtype(self) -> np.dtype:
    return self._wrapped.dtype

  @property
  def bounding_boxes(self) -> list[bounding_box.BoundingBox]:
    return self._wrapped.bounding_boxes


# TODO(timblakely): Replace internal Upsample decorator with this one.
class Upsample(VolumeDecorator):
  """Dynamically upsamples data with nearest neighbors.

  The wrapper behaves as a higher resolution volume.
  """

  scale_zyx: np.ndarray

  def __init__(self, wrapped: base.Volume, scale: array.ArrayLike3d):
    """Initializes the wrapper.

    Args:
      wrapped: Lower resolution volume.
      scale: Integer scale factors as (x, y, z)

    Raises:
      ValueError: If scale is not 3d.
    """
    super().__init__(wrapped)
    if len(scale) != 3:
      raise ValueError('Expected a 3d scale in XYZ format')
    self.scale_zyx = np.array(scale[::-1])

  @property
  def volume_size(self) -> array.Tuple3i:
    return self.scale_zyx[::-1] * self._wrapped.volume_size

  @property
  def voxel_size(self) -> array.Tuple3i:
    return tuple(self._wrapped.pixel_size / self.scale_zyx[::-1])

  @property
  def shape(self) -> array.Tuple4i:
    return np.insert(self.scale_zyx, 0, 1) * self._wrapped.shape

  @property
  def bounding_boxes(self) -> list[bounding_box.BoundingBox]:
    return [b.scale(self.scale_zyx[::-1]) for b in self._wrapped.bounding_boxes]

  def get_points(self, points: array.PointLookups) -> np.ndarray:
    scaled_points = list(points)
    for i in range(1, 4):
      scaled_points[i] = np.array(scaled_points[i]) // self.scale_zyx[i - 1]
    return self._wrapped.get_points(tuple(scaled_points))

  def get_slices(self, slices: array.CanonicalSlice) -> np.ndarray:
    ceildiv = lambda x, y: -(-x // y)

    scaled_slice = list(slices)
    for i in range(1, 4):
      begin = slices[i].start // self.scale_zyx[i - 1]
      assert slices[i].stop is not None
      end = ceildiv(slices[i].stop, self.scale_zyx[i - 1])
      scaled_slice[i] = np.s_[begin:end]
    low_res = self._wrapped.get_slices(tuple(scaled_slice))
    # Upsample.
    upsampled = low_res.repeat(
        self.scale_zyx[0], axis=1).repeat(
            self.scale_zyx[1], axis=2).repeat(
                self.scale_zyx[2], axis=3)
    del low_res
    # Restrict to the requested fragment.
    sel = [slice(None)]
    for i in range(1, 4):
      actual_start = (
          slices[i].start - scaled_slice[i].start * self.scale_zyx[i - 1])
      sel.append(
          slice(actual_start, actual_start + slices[i].stop - slices[i].start))

    return upsampled[tuple(sel)]

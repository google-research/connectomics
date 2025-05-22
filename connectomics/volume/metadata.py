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
"""Metadata for volumetric data."""

import dataclasses
import pathlib
from typing import Sequence

from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.common import tuples
from connectomics.volume import decorators
import dataclasses_json
import numpy as np
import numpy.typing as npt


XYZ = tuples.XYZ


@dataclasses.dataclass(frozen=True)
class VolumeMetadata(
    tuples.DataclassWithNamedTuples,
    dataclasses_json.DataClassJsonMixin,
):
  """Metadata associated with a Volume.

  Attributes:
    volume_size: Volume size in voxels. XYZ order.
    pixel_size: Pixel size in nm. XYZ order.
    bounding_boxes: Bounding boxes associated with the volume.
    num_channels: Number of channels in the volume.
    dtype: Datatype of the volume. Must be numpy compatible.
  """
  path: str
  volume_size: XYZ[int] = tuples.named_tuple_field(XYZ[int])
  pixel_size: XYZ[float] = tuples.named_tuple_field(XYZ[float])
  bounding_boxes: list[bounding_box.BoundingBox] = dataclasses.field(
      default_factory=list
  )
  num_channels: int = 1
  dtype: npt.DTypeLike = dataclasses.field(
      metadata=dataclasses_json.config(
          decoder=np.dtype,
          encoder=lambda x: np.dtype(x).name,
      ),
      default=np.uint8,
  )

  def scale(
      self,
      scale_factors: float | Sequence[float],
      new_path: file.PathLike | None = None,
  ) -> 'VolumeMetadata':
    """Scales the volume metadata by the given scale factors.

    `scale_factors` must be a single float that will be applied multiplicatively
    to the volume size and pixel size, or a 3-element sequence of floats that
    will be applied to XYZ dimensions respectively.

    Args:
      scale_factors: The scale factors to apply.
      new_path: The new path to use for the volume. If None, the original path
        will be used.

    Returns:
      A new VolumeMetadata with the scaled values.
    """
    if isinstance(scale_factors, float) or isinstance(scale_factors, int):
      scale_factors = [scale_factors] * 3
    if len(scale_factors) != 3:
      raise ValueError('scale_factors must be a 3-element sequence.')
    path = new_path if new_path is not None else self.path
    return VolumeMetadata(
        path=path,
        volume_size=tuples.XYZ(*[
            int(x * scale) for x, scale in zip(self.volume_size, scale_factors)
        ]),
        pixel_size=tuples.XYZ(*[
            float(x / scale) for x, scale in zip(self.pixel_size, scale_factors)
        ]),
        bounding_boxes=[
            bbox.scale(scale_factors) for bbox in self.bounding_boxes
        ],
    )

  def scale_xy(
      self, factor: float, new_path: file.PathLike | None = None
  ) -> 'VolumeMetadata':
    return self.scale([factor, factor, 1.0], new_path)


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class DecoratedVolume:
  """A volume with additional metadata.

  Attributes:
    path: The path to the volume.
    decorator_specs: A JSON string of decorator specs, or one or more
      DecoratorSpec objects.
  """

  path: pathlib.Path
  decorator_specs: (
      str | decorators.DecoratorSpec | list[decorators.DecoratorSpec]
  )

  def __post_init__(self):
    if isinstance(self.decorator_specs, list):
      specs = []
      for spec in self.decorator_specs:
        # Support old, internal format.
        if 'decorator' in spec:
          spec['name'] = spec['decorator']
          del spec['decorator']
        if 'kwargs' in spec:
          spec['args'] = decorators.DecoratorArgs.from_dict(spec['kwargs'])
          del spec['kwargs']
        specs.append(decorators.DecoratorSpec.from_dict(spec))
      object.__setattr__(
          self,
          'decorator_specs',
          specs,
      )

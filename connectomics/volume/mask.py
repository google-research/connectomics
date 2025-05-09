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
"""Utilities for dealing with 2d and 3d object masks."""

import dataclasses
import threading
import typing
from typing import Any, Callable, Sequence

from absl import logging
from connectomics.common import bounding_box
from connectomics.volume import metadata
import dataclasses_json
import numpy as np
from scipy import ndimage

# Used to limit memory usage when concurrent requests are processed.
_MASK_SEM = threading.Semaphore(4)


@dataclasses.dataclass
class MaskChannelConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for a single mask channel.

  Specifies how to convert a channel from a TensorStore volume into a Boolean
  exclusion mask. The mask will be formed by:
    min_value <= [channel_value] <= max_value
  or, if `values` is set, to all voxels containing one of the values specified
  therein.

  If `invert` is true, the complement of the mask computed according to the
  above rules is used.
  """
  channel: int
  min_value: float
  max_value: float
  values: list[int] = dataclasses.field(default_factory=list)

  # SECURITY WARNING: This gets passed to Python's eval, which will allow
  # execution of arbitrary code. A Python expression to interpret to form the
  # mask. The mask values are available under 'channel_mask', in addition to
  # 'x', 'y', and 'z' indicating spatial coordinates like in
  # CoordinateExpressionOptions. If specified, will be used instead of
  # min_value/max_value/values.
  expression: str | None = None

  # Value to substitute nan's with.
  nan_value: float | None = None

  # A voxel will be considered masked if any voxels within the FOV centered
  # around it are masked. The FOV start coordinates are relative to current
  # position. If unset, defaults to a FOV of (1, 1, 1), i.e. masked voxels
  # corresponding to the mask exactly.
  fov: bounding_box.BoundingBox | None = None

  invert: bool = False


@dataclasses.dataclass
class VolumeMaskOptions(dataclasses_json.DataClassJsonMixin):
  """Configuration for a volume mask."""
  mask: metadata.DecoratedVolume
  channels: list[MaskChannelConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ImageMaskOptions(dataclasses_json.DataClassJsonMixin):
  """Configuration for an image mask."""
  # SECURITY WARNING: This gets passed to Python's eval, which will allow
  # execution of arbitrary code. This option is for internal use only. The
  # unnormalized 3d image ndarray is accessible under 'image'.
  expression: str | None = None
  channels: list[MaskChannelConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CoordinateExpressionOptions(dataclasses_json.DataClassJsonMixin):
  """Configuration for a coordinate expression."""
  # SECURITY WARNING: This gets passed to Python's eval, which will allow
  # execution of arbitrary code. This option is for internal use only. Valid
  # numpy expression, where 'x', 'y', and 'z' are dense index arrays defining
  # the coordinates of every voxel in the current subvolume, in the global
  # coordinate system of the volume.
  expression: str


@dataclasses.dataclass
class MaskConfig(dataclasses_json.DataClassJsonMixin):
  """Top-level configuration for creating masks."""
  volume: VolumeMaskOptions | None = None
  image: ImageMaskOptions | None = None
  coordinate_expression: CoordinateExpressionOptions | None = None

  invert: bool = False

  def __post_init__(self):
    # Ensure only one is set
    oneof_fields = [self.volume, self.image, self.coordinate_expression]
    if sum([1 if v is not None else 0 for v in oneof_fields]) != 1:
      raise ValueError('Exactly one of volume, image, or '
                       'coordinate_expression must be specified')


@dataclasses.dataclass
class MaskConfigs(dataclasses_json.DataClassJsonMixin):
  masks: list[MaskConfig]
  invert: bool


# TODO(timblakely): Return a Subvolume.
def build_mask(
    masks: Sequence[MaskConfig] | MaskConfigs,
    box: bounding_box.BoundingBox,
    decorated_volume_loader: Callable[[metadata.DecoratedVolume], np.ndarray],
    mask_volume_map: dict[str, Any] | None = None,
    image: np.ndarray | None = None,
    volume_decorator_fn: Callable[[np.ndarray], np.ndarray] = lambda x: x,
) -> np.ndarray:
  """Builds a boolean mask.

  Args:
    masks: iterable of MaskConfig proto or MaskConfigs proto
    box: bounding box defining the area for which to build the mask
    decorated_volume_loader: Function to load a DecoratedVolume object, return
      an ndarray-like object.
    mask_volume_map: optional dict mapping volume proto hashes to open volumes;
      use this as a cache to avoid opening volumes multiple times.
    image: 3d image ndarray; only needed if the mask config uses the image as
      input
    volume_decorator_fn: callable taking a volume object and returning another
      object supporting the same interface.

  Returns:
    boolean mask built according to the specified config

  Raises:
    ValueError: If the mask configuration is unsupported.
  """

  invert = False
  if isinstance(masks, MaskConfigs):
    invert = masks.invert
    masks = masks.masks
  masks = typing.cast(Sequence[MaskConfig], masks)

  final_mask = None
  if mask_volume_map is None:
    mask_volume_map = {}

  z, y, x = np.mgrid[box.to_slice3d()]  # pylint:disable=unused-variable
  mask = None
  for config in masks:
    curr_mask = np.zeros(box.size[::-1], dtype=bool)

    if config.coordinate_expression is not None:
      bool_mask = eval(config.coordinate_expression.expression)  # pylint: disable=eval-used
      curr_mask |= bool_mask
    else:
      if config.image is not None:
        assert image is not None

        if config.image.expression is not None:
          channels = []
          assert not config.image.channels
          curr_mask |= eval(config.image.expression)  # pylint: disable=eval-used
        else:
          channels = config.image.channels
          mask = image[np.newaxis, ...]
      elif config.volume is not None:
        channels = config.volume.channels

        volume_key = config.volume.mask.to_json()
        if volume_key not in mask_volume_map:
          mask_volume_map[volume_key] = volume_decorator_fn(
              decorated_volume_loader(config.volume.mask)
          )
        volume = mask_volume_map[volume_key]
        mask = volume[box.to_slice4d()]
      else:
        logging.fatal('Unsupported mask source: %s', config.to_json())

      for chan_config in channels:
        channel_mask = mask[chan_config.channel, ...]

        if chan_config.nan_value is not None:
          channel_mask = np.nan_to_num(channel_mask, chan_config.nan_value)

        if chan_config.expression:
          bool_mask = eval(chan_config.expression)  # pylint: disable=eval-used
        elif chan_config.values:
          bool_mask = np.isin(channel_mask, chan_config.values).reshape(
              channel_mask.shape
          )
        else:
          assert chan_config.max_value >= chan_config.min_value
          bool_mask = (channel_mask >= chan_config.min_value) & (
              channel_mask <= chan_config.max_value
          )
        if chan_config.invert:
          bool_mask = np.logical_not(bool_mask)

        if chan_config.fov is not None:
          fov = chan_config.fov
          # TODO(timblakely): Type checker appears to be confused without this
          # check...?
          assert fov is not None
          with _MASK_SEM:
            bool_mask = ndimage.maximum_filter(
                bool_mask,
                size=fov.size[::-1],
                origin=fov.size[::-1] // 2 + fov.start[::-1],
            )
        # When using an upsampled mask, the mask could be not the exactly same
        # size as the input image (i.e., -1 offset), we need to pad the mask to
        # match the image size.
        if curr_mask.shape != bool_mask.shape:
          padding = [
              (0, curr_mask.shape[dim] - bool_mask.shape[dim])
              for dim in range(len(curr_mask.shape))
          ]
          bool_mask = np.pad(
              bool_mask, padding, mode='constant', constant_values=0
          )
        curr_mask |= bool_mask

    if config.invert:
      curr_mask = np.logical_not(curr_mask)

    if final_mask is None:
      final_mask = curr_mask
    else:
      final_mask |= curr_mask

  assert final_mask is not None

  if invert:
    return np.logical_not(final_mask)
  else:
    return final_mask

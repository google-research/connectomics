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
"""4-D volume abstraction."""

import typing
from typing import List

from connectomics.common import array
from connectomics.common import bounding_box
import numpy as np


# TODO(timblakely): Make generic-typed so it exposes both VolumeInfo and
# Tensorstore via .descriptor.
class BaseVolume:
  """Common interface to multiple volume backends for Decorators."""

  # TODO(timblakely): Convert to returning Subvolumes.
  def __getitem__(self, ind: array.IndexExpOrPointLookups) -> np.ndarray:
    ind = array.normalize_index(ind, self.shape)

    if array.is_point_lookup(ind):
      # Hack to make pytype happy. We've taken care of checking for the
      # point-lookup path in the above conditional
      ind = typing.cast(array.PointLookups, ind)
      return self.get_points(ind)
    return self.get_slices(ind)

  # TODO(timblakely): Remove any usage of this with Vector3j.
  def get_points(self, points: array.PointLookups) -> np.ndarray:
    """Returns values at points given `channel, list[X], list[Y], list[Z]`."""
    raise NotImplementedError

  # TODO(timblakely): Convert to returning Subvolumes.
  def get_slices(self, slices: array.CanonicalSlice) -> np.ndarray:
    """Returns a subvolume of data based on a specified set of CZYX slices."""
    raise NotImplementedError

  @property
  def volume_size(self) -> array.Tuple3i:
    """Volume size in voxels, XYZ."""
    raise NotImplementedError

  @property
  def voxel_size(self) -> array.Tuple3f:
    """Size of an individual voxels in physical dimensions (Nanometers)."""
    raise NotImplementedError

  @property
  def shape(self) -> array.Tuple4i:
    """Shape of the volume in voxels, CZYX."""
    raise NotImplementedError

  @property
  def ndim(self) -> int:
    """Number of dimensions in this volume."""
    raise NotImplementedError

  @property
  def dtype(self) -> np.dtype:
    """Datatype of the underlying data."""
    raise NotImplementedError

  @property
  def bounding_boxes(self) -> List[bounding_box.BoundingBox]:
    """List of bounding boxes contained in this volume."""
    raise NotImplementedError

  # TODO(timblakely): determine what other attributes we want to make mandatory for
  # all implementations and add them here.

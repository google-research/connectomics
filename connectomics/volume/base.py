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
from typing import Union

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.volume import subvolume
import numpy as np

Subvolume = subvolume.Subvolume


def slice_to_bbox(ind: array.CanonicalSlice) -> bounding_box.BoundingBox:
  rel_start, rel_end = tuple(zip(*[(i.start, i.stop) for i in ind[3:0:-1]]))
  return bounding_box.BoundingBox(start=rel_start, end=rel_end)


class VolumeIndexer:
  """Interface for indexing supporting point lookups and slices."""
  _volume: 'BaseVolume'
  slices: array.CanonicalSlice

  def __init__(self, volume: 'BaseVolume'):
    self._volume = volume

  def __getitem__(
      self, ind: array.IndexExpOrPointLookups
  ) -> Union[np.ndarray, subvolume.Subvolume]:
    """Returns the results of a point lookup or the slices and sliced data."""
    ind = array.normalize_index(ind, self._volume.shape)

    if array.is_point_lookup(ind):
      # Hack to make pytype happy. We've taken care of checking for the
      # point-lookup path in the above conditional
      ind = typing.cast(array.PointLookups, ind)
      return self._volume.get_points(ind)
    return subvolume.Subvolume(self._volume.get_slices(ind), slice_to_bbox(ind))


class DirectVolumeIndexer(VolumeIndexer):
  """VolumeIndexer that ignores whether access was point lookup or slices."""

  def __getitem__(self, ind: array.IndexExpOrPointLookups) -> np.ndarray:
    result = super().__getitem__(ind)
    return result if isinstance(result, np.ndarray) else result.data


# TODO(timblakely): Make generic-typed so it exposes both VolumeInfo and
# Tensorstore via .descriptor.
class BaseVolume:
  """Common interface to multiple volume backends for Decorators."""

  def __getitem__(
      self, ind: array.IndexExpOrPointLookups) -> Union[np.ndarray, Subvolume]:
    return VolumeIndexer(self)[ind]

  def __setitem__(self, ind: array.IndexExpOrPointLookups, value: np.ndarray):
    ind = array.normalize_index(ind, self.shape)
    if array.is_point_lookup(ind):
      self.write_points(ind, value)
    else:
      self.write_slices(ind, value)

  # TODO(timblakely): Only a temporary shim while we convert all internal usage to
  # using Subvolumes.
  @property
  def asarray(self) -> DirectVolumeIndexer:
    """__getitem__-like indexing that only returns the bare ndarray."""
    return DirectVolumeIndexer(self)

  # TODO(timblakely): Remove any usage of this with Vector3j.
  def get_points(self, points: array.PointLookups) -> np.ndarray:
    """Returns values at points given `channel, list[X], list[Y], list[Z]`."""
    raise NotImplementedError

  def get_slices(self, slices: array.CanonicalSlice) -> np.ndarray:
    """Returns a subvolume of data based on a specified set of CZYX slices."""
    raise NotImplementedError

  def write_points(self, points: array.PointLookups, values: np.ndarray):
    """Writes values at points given `channel, list[X], list[Y], list[Z]`."""
    raise NotImplementedError

  def write_slices(self, slices: array.CanonicalSlice, value: np.ndarray):
    """Writes a subvolume of data based on a specified set of CZYX slices."""
    raise NotImplementedError

  def write(self, subvol: subvolume.Subvolume):
    self.write_slices(
        array.normalize_index(subvol.bbox.to_slice4d(), self.shape),
        subvol.data)

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
  def bounding_boxes(self) -> list[bounding_box.BoundingBox]:
    """List of bounding boxes contained in this volume."""
    raise NotImplementedError

  @property
  def chunk_size(self) -> array.Tuple4i:
    """Backing chunk size in voxels, CZYX."""
    raise NotImplementedError

  def clip_box_to_volume(
      self, box: bounding_box.BoundingBox) -> bounding_box.BoundingBox:
    return bounding_box.BoundingBox(
        box.start, end=np.minimum(box.end, self.volume_size))

  # TODO(timblakely): determine what other attributes we want to make mandatory for
  # all implementations and add them here.

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
"""A Tensorstore-backed Volume."""

import dataclasses
import functools
from typing import Any, Sequence, Union

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.volume import base
import dataclasses_json
import numpy as np

import tensorstore as ts


def tuple_deserialize(v: Sequence[Union[int, float]]) -> array.Tuple3f:
  if isinstance(v, tuple):
    return v
  return tuple(v)


TensorstoreSpec = Union[str, dict[str, Any]]


@dataclasses_json.dataclass_json
@functools.partial(dataclasses.dataclass, eq=True)
class TensorstoreMetadata:
  """Additional volumetric metadata associated with TensorStore volumes.

  Attributes:
    voxel_size: Voxel size in nm.
    bounding_boxes: Bounding boxes associated with this tensorstore.
  """
  voxel_size: array.Tuple3f = dataclasses.field(
      metadata=dataclasses_json.config(decoder=tuple_deserialize))
  bounding_boxes: list[bounding_box.BoundingBox] = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda bboxes: [b.spec for b in bboxes],
          decoder=lambda bboxes: [bounding_box.deserialize(b) for b in bboxes]))

  def __post_init__(self):
    # Purely to ensure that voxel_size is a tuple if initialized with a list.
    self.voxel_size = tuple(self.voxel_size)


@dataclasses_json.dataclass_json
@functools.partial(dataclasses.dataclass, eq=True)
class TensorstoreConfig:
  spec: TensorstoreSpec
  metadata: TensorstoreMetadata = dataclasses.field(
      metadata=dataclasses_json.config(
          decoder=file.dataclass_loader(TensorstoreMetadata)))


class TensorstoreVolume(base.BaseVolume):
  """Tensorstore-backed Volume."""

  _store: ts.TensorStore
  _config: TensorstoreConfig

  def __init__(self, config: TensorstoreConfig):
    if not config.metadata.bounding_boxes:
      raise ValueError('Config must have at least one bounding box')
    if not config.metadata.voxel_size or any(
        [v <= 0 for v in config.metadata.voxel_size]):
      raise ValueError(f'Invalid voxel size: {config.metadata.voxel_size}')

    self._config = config
    store = ts.open(config.spec).result()

    if store.ndim != 4:
      raise ValueError(f'Expected tensorstore to be 4D, found: {store.ndim}')
    valid_sizes = np.all([
        store.shape[3:0:-1] <= bbox.end
        for bbox in config.metadata.bounding_boxes
    ])

    if not valid_sizes:
      raise ValueError(
          'TensorStore volume extends beyond all known bounding boxes')
    self._store = store

  def get_points(self, points: array.PointLookups) -> np.ndarray:
    return self._store[points].read().result()

  def get_slices(self, slices: array.CanonicalSlice) -> np.ndarray:
    return self._store[slices].read().result()

  @property
  def volume_size(self) -> array.Tuple3i:
    # TODO(timblakely): Handle non-CZYX volumes.
    return tuple(self._store.shape[3:0:-1])

  @property
  def voxel_size(self) -> array.Tuple3f:
    return self._config.metadata.voxel_size

  @property
  def shape(self) -> array.Tuple4i:
    return self._store.shape

  @property
  def ndim(self) -> int:
    return len(self._store.shape)

  @property
  def dtype(self) -> np.dtype:
    return self._store.dtype

  @property
  def bounding_boxes(self) -> list[bounding_box.BoundingBox]:
    return self._config.metadata.bounding_boxes

  @property
  def metadata(self) -> TensorstoreMetadata:
    return self._config.metadata

  def write_slices(self, slices: array.CanonicalSlice, value: np.ndarray):
    """Writes a subvolume of data based on a specified set of CZYX slices."""
    with ts.Transaction():
      self._store[slices].write(value).result()


class TensorstoreArrayVolume(TensorstoreVolume):
  """TensorStore volume using existing, in-memory arrays."""

  def __init__(self, data: np.ndarray, metadata: Union[str,
                                                       TensorstoreMetadata]):
    config = TensorstoreConfig(
        spec={
            'driver': 'array',
            'dtype': str(data.dtype),
            'array': data,
        },
        metadata=metadata)
    super().__init__(config)

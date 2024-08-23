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

import copy
import dataclasses
from typing import Any, Optional, Sequence, Union

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.common import utils
from connectomics.volume import base
import dataclasses_json
import numpy as np
import tensorstore as ts


def tuple_deserialize(v: Sequence[Union[int, float]]) -> array.Tuple3f:
  if isinstance(v, tuple):
    return v
  return tuple(v)


@dataclasses.dataclass(eq=True)
class TensorstoreMetadata(utils.NPDataClassJsonMixin):
  """Additional volumetric metadata associated with TensorStore volumes.

  Attributes:
    voxel_size: Voxel size in nm.
    bounding_boxes: Bounding boxes associated with this tensorstore.
  """
  voxel_size: array.Tuple3f
  bounding_boxes: list[bounding_box.BoundingBox]

  def __post_init__(self):
    # Purely to ensure that voxel_size is a tuple if initialized with a list.
    self.voxel_size = tuple(self.voxel_size)


@dataclasses.dataclass(eq=True)
class TensorstoreConfig(utils.NPDataClassJsonMixin):
  spec: dict[str, Any]
  metadata: Optional[TensorstoreMetadata] = dataclasses.field(
      default=None,
      metadata=dataclasses_json.config(
          decoder=file.dataclass_loader(TensorstoreMetadata)))


class TensorstoreVolume(base.Volume):
  """Tensorstore-backed Volume."""

  _store: ts.TensorStore
  _config: TensorstoreConfig

  def __init__(self, config: TensorstoreConfig):
    assert config.metadata
    if not config.metadata.bounding_boxes:
      raise ValueError('Config must have at least one bounding box')
    if not config.metadata.voxel_size or any(
        [v <= 0 for v in config.metadata.voxel_size]):
      raise ValueError(f'Invalid voxel size: {config.metadata.voxel_size}')

    self._config = config
    store = ts.open(config.spec).result()

    if store.ndim != 4:
      raise ValueError(f'Expected tensorstore to be 4D, found: {store.ndim}')

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
    assert self._config.metadata
    return self._config.metadata.voxel_size

  @property
  def shape(self) -> array.Tuple4i:
    return self._store.shape

  @property
  def ndim(self) -> int:
    return len(self._store.shape)

  @property
  def dtype(self) -> np.dtype:
    return self._store.dtype.numpy_dtype

  @property
  def bounding_boxes(self) -> list[bounding_box.BoundingBox]:
    assert self._config.metadata
    return self._config.metadata.bounding_boxes

  @property
  def metadata(self) -> TensorstoreMetadata:
    assert self._config.metadata
    return self._config.metadata

  def write_slices(self,
                   slices: array.CanonicalSlice,
                   value: np.ndarray,
                   use_transaction=True):
    """Writes a subvolume of data based on a specified set of CZYX slices."""

    def _write():
      self._store[tuple(slices)].write(value).result()

    if use_transaction:
      with ts.Transaction():
        _write()
    else:
      _write()

  @property
  def chunk_size(self) -> array.Tuple4i:
    """Backing chunk size in voxels, CZYX."""
    return self._store.schema.chunk_layout.read_chunk.shape


class TensorstoreArrayVolume(TensorstoreVolume):
  """TensorStore volume using existing, in-memory arrays."""

  def __init__(self, data: np.ndarray, metadata: TensorstoreMetadata):
    config = TensorstoreConfig(
        spec={
            'driver': 'array',
            'dtype': str(data.dtype),
            'array': data,
        },
        metadata=metadata)
    super().__init__(config)


def create_volume_if_necessary(tensorstore_config: TensorstoreConfig):
  tensorstore_config = copy.deepcopy(tensorstore_config)
  tensorstore_config.spec['create'] = True
  tensorstore_config.spec['delete_existing'] = True
  # Open happens in constructor.
  _ = TensorstoreVolume(tensorstore_config)


def load_ts_config(spec: Union[str, TensorstoreConfig]) -> TensorstoreConfig:
  config = file.load_dataclass(TensorstoreConfig, spec)
  if config is None:
    raise ValueError(f'Could not load descriptor: {spec}')
  return config



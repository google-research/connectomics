"""A Tensorstore-backed Volume."""

import dataclasses
import json
import typing
from typing import Any, Sequence, Union

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.volume import base
import dataclasses_json
import numpy as np

import tensorstore as ts


def tuple_deserialize(v: Sequence[Union[int, float]]) -> array.Tuple3f:
  if isinstance(v, tuple):
    return v
  return tuple(v)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TensorstoreVolumeMetadata:
  voxel_size: array.Tuple3f = dataclasses.field(
      metadata=dataclasses_json.config(decoder=tuple_deserialize))
  bounding_boxes: list[bounding_box.BoundingBox] = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda bboxes: [b.spec for b in bboxes],
          decoder=lambda bboxes: [bounding_box.deserialize(b) for b in bboxes]))


class TensorstoreVolume(base.BaseVolume):
  """Tensorstore-backed Volume."""

  _store: ts.TensorStore
  _metadata: TensorstoreVolumeMetadata

  def __init__(self, tensorstore_spec: Union[str, dict[str, Any]],
               metadata: Union[str, TensorstoreVolumeMetadata]):

    if isinstance(tensorstore_spec, str):
      tensorstore_spec = json.loads(tensorstore_spec)
    # Casting is okay here since we've converted from str to dict[str, Any]
    # above.
    tensorstore_spec = typing.cast(dict[str, Any], tensorstore_spec)

    if isinstance(metadata, str):
      # Try and deserialize it first, since that's usually faster than file
      # operations. If that fails, assume it's a file path.
      try:
        self._metadata = TensorstoreVolumeMetadata.from_json(metadata)
      except json.decoder.JSONDecodeError:
        # Try and load it as a file path instead.
        try:
          # TODO(timblakely): Shim this so we can work with internal file systems.
          with open(metadata, 'r') as f:
            # TODO(timblakely): Due to
            # https://github.com/lidatong/dataclasses-json/issues/318 we can't
            # use the spec(), which in turn means we can't use dataclass.load or
            # .loads. Instead we just read the contents directly and attempt to
            # parse it as json.
            self._metadata = TensorstoreVolumeMetadata.from_json(f.read())
        except FileNotFoundError:
          raise ValueError(
              'Could not parse metadata json, or file does not exist: '
              f'{metadata}'
          ) from None
    else:
      self._metadata = metadata
    self._store = ts.open(tensorstore_spec).result()

  def get_points(self, points: array.PointLookups) -> np.ndarray:
    return self._store[points].read().result()

  # TODO(timblakely): Convert to returning Subvolumes.
  def get_slices(self, slices: array.CanonicalSlice) -> np.ndarray:
    return self._store[slices].read().result()

  @property
  def volume_size(self) -> array.Tuple3i:
    # TODO(timblakely): Handle non-CZYX volumes.
    return tuple(self._store.shape[3:0:-1])

  @property
  def voxel_size(self) -> array.Tuple3f:
    return self._metadata.voxel_size

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
    return self._metadata.bounding_boxes

  @property
  def metadata(self) -> TensorstoreVolumeMetadata:
    return self._metadata


class TensorstoreArrayVolume(TensorstoreVolume):
  """TensorStore volume using existing, in-memory arrays."""

  def __init__(
      self,
      data: np.ndarray,
      metadata: Union[str, TensorstoreVolumeMetadata]):
    super().__init__(
        {
            'driver': 'array',
            'dtype': str(data.dtype),
            'array': data,
        }, metadata)

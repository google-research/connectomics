# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Shim for supporting internal and external file IO."""

import dataclasses
import functools
import json
import pathlib
import typing
from typing import Any, Callable, Type, TypeVar, Union
import urllib.parse

from absl import logging
import dataclasses_json
import tensorstore as ts

PathLike = Union[str, pathlib.PurePath]

T = TypeVar('T', bound=dataclasses_json.DataClassJsonMixin)

# Local Alias
Path = pathlib.Path


def save_dataclass_json(
    dataclass_instance: T,
    path: PathLike,
    json_path: str | None = None,
    kvdriver: str = 'file',
    kvstore: dict[str, Any] | None = None,
):
  """Save a dataclass to a file.

  Args:
    dataclass_instance: Dataclass to save.
    path: Path to save to.
    json_path: Optional path to save to within the file.
    kvdriver: Driver to use for saving when using the file driver.
    kvstore: Optional override for the kvstore to pass to TensorStore.
  """
  if kvstore is None:
    kvstore = {'driver': kvdriver, 'path': str(path)}
  spec = {
      'driver': 'json',
      'kvstore': kvstore,
  }
  if json_path is not None:
    if not json_path.startswith('/'):
      json_path = f'/{json_path}'
    spec['json_pointer'] = json_path
  meta_ts = ts.open(spec).result()
  meta_ts.write(dataclass_instance.to_dict()).result()


@dataclasses.dataclass
class TensorStoreDataSourceAdapter:
  """TensorStore data source adapter.

  Attributes:
    name: Name of the adapter type.
    param: Parameter to the adapter.
  """

  name: str
  param: str


@dataclasses.dataclass
class TensorStoreDataSource:
  """TensorStore data source, following the Neuroglancer data source format.

  Attributes:
    name: Name of the source.
    path: Path to the source.
    adapters: List of adapters applied to the source.
    uri: TensorStore data source URI. Format is <source>(|<optional-adapters>),
      or using TensorStore terms <kvstore-url>(|<optional-adapters>).
      Example: gs://my-bucket/path/to/volume.zarr.zip|zip:path/to/entry
  """

  name: str
  path: str
  adapters: list['TensorStoreDataFormat'] = dataclasses.field(
      default_factory=list
  )

  @property
  def uri(self) -> str:
    return f'{self.name}:{self.path}'


@dataclasses.dataclass
class TensorStoreDataFormat:
  """TensorStore data format adapter.

  Attributes:
    driver: Name of the format driver.
    param: Additional parameters passed to the driver.
  """

  driver: str
  param: str | None = None


@dataclasses.dataclass
class TensorStorePath:
  """TensorStore data source, following the Neuroglancer data source format.

  Source: https://neuroglancer-docs.web.app/datasource/index.html

  Attributes:
    uri: TensorStore path. Format is <source>(|<optional-adapters>)|<format>:,
      or using TensorStore terms <kvstore-url>(|<optional-adapters>)|<driver>:.
      Example: gs://my-bucket/path/to/volume.zarr.zip|zip:path/to/entry|zarr:
  """

  uri: str

  def open_spec(self, kvdriver='file') -> dict[str, Any]:
    """Returns a TensorStore spec that can be used to open the TensorStore."""
    # TODO(timblakely): Support other formats.
    spec = {}

    # Special case
    match self.format.driver:
      case 'neuroglancer_precomputed' | 'neuroglancer-precomputed':
        spec['driver'] = 'neuroglancer_precomputed'
      case 'volumestore' | 'volinfo':
        # Note
        # that Volumestore does not support adapters, so we can safely return
        # early here.
        path = self.source.path
        if not path.endswith('.volinfo'):
          path = f'{path}.volinfo'
        spec['driver'] = 'volumestore'
        spec['volinfo_path'] = path
        return spec
      case _:
        raise ValueError(
            f'Unknown TensorStore format driver: {self.format.driver}'
        )

    if self.source.name == 'file':
      spec['kvstore'] = f'{kvdriver}://{self.source.path}'
    else:
      spec['kvstore'] = f'{self.source.name}://{self.source.path}'

    if not self.adapters:
      return spec

    # There are adapters, so we need to add them to the spec.
    base_spec = {'kvstore': spec['kvstore']}
    for adapter in self.adapters:
      adapter_spec = {
          'driver': adapter.name,
          'path': adapter.param,
          'base': base_spec,
      }
      base_spec = adapter_spec
    spec['kvstore'] = base_spec
    return spec

  @property
  def source(self) -> TensorStoreDataSource:
    """Returns the source (kvstore-url) of the TensorStore."""
    return TensorStoreDataSource(*self._parts[0].split('://', 1), self.adapters)

  @property
  def adapters(self) -> list[TensorStoreDataSourceAdapter]:
    """Returns the adapters (if any) applied to the source kvstore."""
    return [
        TensorStoreDataSourceAdapter(*a.split(':', 1))
        for a in self._parts[1:-1]
    ]

  @property
  def format(self) -> TensorStoreDataFormat:
    """Returns the storage format (driver) of the TensorStore."""
    driver, param = self._parts[-1].split(':', 1)
    driver = driver.replace('-', '_')
    return TensorStoreDataFormat(driver, param)

  @property
  def _parts(self) -> list[str]:
    return self.uri.split('|')

  @classmethod
  def from_tensorstore(cls, vol: ts.TensorStore) -> 'TensorStorePath':
    # TODO(timblakely): Support adapters when TensorStore does.
    path = f'{vol.spec().kvstore.url}|{vol.spec().to_json()["driver"]}:'
    return cls(path)

  def __post_init__(self):
    if self.uri.startswith('/'):
      safe_uri = urllib.parse.quote(self.uri, safe=':|/\\')
      self.uri = f'file://{safe_uri}'
    if len(self._parts) < 2:
      raise ValueError(
          'TensorStore URI must contain at least 2 parts: <source>|<format>.'
          ' Auto-detection of format not yet supported.'
      )
    # Double-check that the source kvstore is backed by gfile, as
    # Volumestore's TensorStore driver depends on sstables via gfile.
    if self.format.driver == 'volumestore' and not self.source.name.startswith(
        'gfile'
    ):
      raise ValueError('Volumestore TensorStore driver requires gfile kvstore.')


def dataclass_from_serialized(
    target: Type[T],
    serialized: Union[str, PathLike],
    infer_missing_fields: bool = False,
) -> T:
  """Load a dataclass from a serialized instance, file path, or dict.

  Args:
    target: Dataclass to load
    serialized: Serialized instance, file path, or dict to create dataclass
      from.
    infer_missing_fields: Whether to infer missing fields.

  Returns:
    New dataclass instance.
  """
  as_str = str(serialized)
  # Try to load the dataclass directly
  if not as_str.startswith('@'):
    try:
      return target.from_json(serialized)
    except json.JSONDecodeError:
      logging.warning(
          'Could not decode %s as JSON %s, trying to load as a path',
          serialized,
          target,
      )
  if as_str.startswith('@'):
    as_str = as_str[1:]
  return load_dataclass_json(
      target,
      as_str,
      infer_missing_fields=infer_missing_fields,
  )


def load_json(
    json_or_path: str | PathLike,
    json_path: str | None = None,
) -> dict[str, Any]:
  """Load a JSON object from a string or file path via TensorStore."""
  try:
    return json.loads(json_or_path)
  except json.JSONDecodeError:
    logging.warning(
        'Could not decode %s as JSON, trying to load as a path', json_or_path
    )
  path = json_or_path
  spec = {
      'driver': 'json',
      'kvstore': str(path),
  }
  if json_path is not None:
    if not json_path.startswith('/'):
      json_path = f'/{json_path}'
    spec['json_pointer'] = json_path
  return ts.open(spec).result().read().result().item()


def load_dataclass_json(
    dataclass_type: Type[T],
    path: PathLike,
    json_path: str | None = None,
    infer_missing_fields: bool = False,
) -> T:
  """Load a dataclass from a file path.

  Args:
    dataclass_type: Dataclass to load
    path: Path to load from.
    json_path: Optional path to load from within the file.
    infer_missing_fields: Whether to infer missing fields.

  Returns:
    New dataclass instance.
  """
  return dataclass_type.from_dict(
      load_json(path, json_path),
      infer_missing=infer_missing_fields,
  )


# TODO(timblakely): Remove in favor the above serialization.
def load_dataclass(
    constructor: type[T], v: Union[str, dict[str, Any], T, None]
) -> Union[T, None]:
  """Load a dataclass from a serialized instance, file path, or dict.

  Args:
    constructor: Dataclass to load
    v: Serialized instance, file path, or dict to create dataclass from.

  Returns:
    New dataclass instance.
  """
  if isinstance(v, type(constructor)):
    return typing.cast(T, v)
  elif v is None:
    return v
  elif isinstance(v, str):
    try:
      # We attempt to parse first since file open ops can be expensive.
      return constructor.from_json(v)
    except json.JSONDecodeError:
      # File path; attempt to load.
      with Path(v).open() as f:
        return constructor.from_json(f.read())
  else:
    return constructor.from_dict(typing.cast(dict[str, Any], v))


def dataclass_loader(
    constructor: type[T],
) -> Callable[[Union[str, dict[str, Any], T, None]], Union[T, None]]:
  """Create a dataclass instance from a serialized instance, file path, or dict.

  Args:
    constructor: Constructor of class to instantiate.

  Returns:
    A callable decoder that takes a path to file containing serialized
    dataclass, or dict containing all fields of a dataclass and returns an
    instance of that class.
  """
  return functools.partial(load_dataclass, constructor)

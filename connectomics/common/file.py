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
"""Shim for supporting internal and external file IO."""

from __future__ import annotations

import functools
import json
import pathlib
import typing
from typing import Any, Callable, Optional, Type, TypeVar, Union

from absl import logging
import dataclasses_json
import tensorstore as ts


T = TypeVar('T', bound=dataclasses_json.DataClassJsonMixin)
PathLike = Union[str, pathlib.Path]


def save_dataclass_json(
    dataclass_instance: T,
    path: PathLike,
    json_path: Optional[str] = None,
    kvdriver: str = 'file',
):
  """Save a dataclass to a file.

  Args:
    dataclass_instance: Dataclass to save.
    path: Path to save to.
    json_path: Optional path to save to within the file.
    kvdriver: Driver to use for saving.
  """
  spec = {
      'driver': 'json',
      'kvstore': {'driver': kvdriver, 'path': str(path)},
  }
  if json_path is not None:
    if not json_path.startswith('/'):
      json_path = f'/{json_path}'
    spec['json_pointer'] = json_path
  meta_ts = ts.open(spec).result()
  meta_ts.write(dataclass_instance.to_dict()).result()


def dataclass_from_serialized(
    target: Type[T],
    serialized: Union[str, PathLike],
    kvdriver: str = 'file',
    infer_missing_fields: bool = False,
) -> T:
  """Load a dataclass from a serialized instance, file path, or dict.

  Args:
    target: Dataclass to load
    serialized: Serialized instance, file path, or dict to create dataclass
      from.
    kvdriver: Driver to use for loading.
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
      kvdriver=kvdriver,
      infer_missing_fields=infer_missing_fields,
  )


def load_dataclass_json(
    dataclass_type: Type[T],
    path: PathLike,
    json_path: Optional[str] = None,
    kvdriver: str = 'file',
    infer_missing_fields: bool = False,
) -> T:
  """Load a dataclass from a file path.

  Args:
    dataclass_type: Dataclass to load
    path: Path to load from.
    json_path: Optional path to load from within the file.
    kvdriver: Driver to use for loading.
    infer_missing_fields: Whether to infer missing fields.

  Returns:
    New dataclass instance.
  """
  spec = {
      'driver': 'json',
      'kvstore': {'driver': kvdriver, 'path': str(path)},
  }
  if json_path is not None:
    if not json_path.startswith('/'):
      json_path = f'/{json_path}'
    spec['json_pointer'] = json_path
  return dataclass_type.from_dict(
      ts.open(spec).result().read().result().item(),
      infer_missing=infer_missing_fields,
  )


# TODO(timblakely): Remove dependency on TF when there's a common API to read
# files internally and externally.
import tensorflow as tf

Copy = tf.io.gfile.copy
DeleteRecursively = tf.io.gfile.rmtree
Exists = tf.io.gfile.exists
Glob = tf.io.gfile.glob
IsDirectory = tf.io.gfile.isdir
ListDirectory = tf.io.gfile.listdir
MakeDirs = tf.io.gfile.makedirs
MkDir = tf.io.gfile.mkdir
Open = tf.io.gfile.GFile
Remove = tf.io.gfile.remove
Rename = tf.io.gfile.rename
Stat = tf.io.gfile.stat
Walk = tf.io.gfile.walk

GFile = tf.io.gfile.GFile

NotFoundError = tf.errors.NotFoundError


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
      with Open(v) as f:
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

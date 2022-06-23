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

import functools
import json
from typing import Any, Callable, TypeVar, Union

# TODO(timblakely): Remove dependency on TF when there's a common API to read
# files internally and externally.
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.io import gfile

Copy = gfile.copy
DeleteRecursively = gfile.rmtree
Exists = gfile.exists
Glob = gfile.glob
IsDirectory = gfile.isdir
ListDirectory = gfile.listdir
MakeDirs = gfile.makedirs
MkDir = gfile.mkdir
Open = gfile.GFile
Remove = gfile.remove
Rename = gfile.rename
Stat = gfile.stat
Walk = gfile.walk

NotFoundError = tf.errors.NotFoundError

GFile = gfile.GFile

T = TypeVar('T')


def load_dataclass(constructor: T, v: Union[str, dict[str, Any], T, None]) -> T:
  """Load a dataclass from a serialized instance, file path, or dict.

  Args:
    constructor: Dataclass to load
    v: Serialized instance, file path, or dict to create dataclass from.

  Returns:
    New dataclass instance.
  """
  if isinstance(v, type(T)) or v is None:
    return v
  elif isinstance(v, str):
    try:
      # We attempt to parse first since file open ops can be expensive.
      return constructor.from_json(v)
    except json.JSONDecodeError:
      # File path; attempt to load.
      with Open(v) as f:
        return constructor.from_json(f.read())
  return constructor.from_dict(v)


def dataclass_loader(
    constructor: T) -> Callable[[Union[str, dict[str, Any], T, None]], T]:
  """Create a dataclass instance from a serialized instance, file path, or dict.

  Args:
    constructor: Constructor of class to instantiate.

  Returns:
    A callable decoder that takes a path to file containing serialized
    dataclass, or dict containing all fields of a dataclass and returns an
    instance of that class.
  """
  return functools.partial(load_dataclass, constructor)

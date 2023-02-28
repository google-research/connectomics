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
import typing
from typing import Any, Callable, Dict, TypeVar, Union

import dataclasses_json

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

T = TypeVar('T', bound=dataclasses_json.DataClassJsonMixin)


def load_dataclass(constructor: type[T], v: Union[str, dict[str, Any], T,
                                                  None]) -> Union[T, None]:
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
    return constructor.from_dict(typing.cast(Dict[str, Any], v))


def dataclass_loader(
    constructor: type[T]
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

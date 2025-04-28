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
"""Various utility functions."""

import contextlib
import dataclasses
import io
import itertools
import re
import time
from typing import Any, Protocol, TypeVar
import zlib

from absl import logging
import dataclasses_json
import numpy as np


def batch(iterable, n):
  """Chunks an iterable into pieces of 'n' elements or less.

  Args:
    iterable: iterable to chunk
    n: number of elements per chunk

  Yields:
    lists of at most 'n' items from seq
  """
  it = iter(iterable)
  chunk = list(itertools.islice(it, n))
  while chunk:
    yield chunk
    chunk = list(itertools.islice(it, n))


def is_intlike(v) -> bool:
  return isinstance(v,
                    (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                     np.int64, np.uint8, np.uint16, np.uint32, np.uint64, int))


def is_floatlike(v) -> bool:
  return isinstance(v, (np.float16, np.float32, np.float64, float))


def is_boollike(v) -> bool:
  return isinstance(v, (bool, np.bool_))


def from_np_type(v):
  if is_intlike(v):
    return int(v)
  elif is_floatlike(v):
    return float(v)
  elif is_boollike(v):
    return bool(v)
  raise ValueError(f'Unexpected limit type: {v} ({type(v)})')


def _handle_np(o):
  if hasattr(o, 'item'):
    return o.item()
  raise TypeError


class NPDataClassJsonMixin(dataclasses_json.DataClassJsonMixin):
  """Dataclass mixin that supports Numpy decimal types e.g. np.int64.

  Occasionally dataclasses are assigned values that are wrapped in a numpy type.
  If this happens, the built-in JSON encoder used by dataclasses-json cannot do
  the appropriate coersion to the underlying numpy type automatically. This shim
  allows for converting from the various numpy types to the underlying python
  type before deferring to the original encoder.
  """

  def to_json(self, *args, **kw) -> str:
    return super().to_json(*args, default=_handle_np, **kw)


@contextlib.contextmanager
def report_time(name):
  start = time.time()
  try:
    yield
  finally:
    duration = time.time() - start
    logging.info('time[%s] = %.6f', name, duration)


_PASCAL_TO_KEBAB_RE = re.compile(
    r"""
    (?<=[a-z])(?=[A-Z])
    |
    (?<=[A-Z])(?=[A-Z][a-z])
    """,
    re.X,
)


def pascal_to_kebab(name: str) -> str:
  """Converts a PascalCase name to kebab-case."""
  return _PASCAL_TO_KEBAB_RE.sub('-', name).lower()


# This is a kludge, since dataclasses do not export their typeshed for dataclass
# externally.
class IsDataclass(Protocol):
  __dataclass_fields__: dict[str, Any]


D = TypeVar('D', bound=IsDataclass)


# TODO(timblakely): Move to a more appropriate location.
# TODO(timblakely): Support Sequences of Dataclasses.
def update_dataclass(
    source: D,
    overrides: dict[str, Any],
    apply_recursive: bool = True,
) -> D:
  """Recursively updates a dataclass with overrides.

  Contrary to dataclasses.replace, this function will only update attributes
  that are present in the source dataclass, and will apply recursively to
  sub-dataclasses.

  Args:
    source: The dataclass to update.
    overrides: A mapping of attribute name to value to override.
    apply_recursive: Whether to apply the overrides recursively to
      sub-dataclasses or not.

  Returns:
    A new dataclass with the overrides applied.
  """
  params = {}
  for k, v in overrides.items():
    if not hasattr(source, k):
      raise ValueError(f'Attribute {k} not found in {source}')
    attr = getattr(source, k)
    if dataclasses.is_dataclass(attr) and apply_recursive:
      params[k] = update_dataclass(attr, v)
    else:
      params[k] = v

  source = dataclasses.replace(source, **params)
  return source


def serialize_array(
    array: np.ndarray, compression: int | None = None) -> bytes:
  """Serializes Numpy array.

  Args:
    array: Array to serialize.
    compression: Optional compression; `zlib.compress` level.

  Returns:
    (Compressed) serialized array.
  """
  buffer = io.BytesIO()
  np.save(buffer, np.asarray(array), allow_pickle=False)
  if compression is None:
    return buffer.getvalue()
  else:
    return zlib.compress(buffer.getvalue(), level=compression)


def deserialize_array(array: bytes, decompress: bool = False) -> np.ndarray:
  """Deserializes Array to NumPy.

  Args:
    array: (Compressed) serialized array obtained with `serialize_array`.
    decompress: If True, decompresses with zlib.

  Returns:
    Deserialized array.
  """
  buffer = io.BytesIO()
  if not decompress:
    buffer.write(array)
  else:
    buffer.write(zlib.decompress(array))
  buffer.seek(0)
  return np.load(buffer)

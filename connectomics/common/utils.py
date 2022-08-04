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

import itertools

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
  return isinstance(v, (np.float_, np.float16, np.float32, np.float64, float))


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

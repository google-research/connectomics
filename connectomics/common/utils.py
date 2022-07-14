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
from typing import Any

import numpy as np
import tensorstore as ts


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


_TS_DTYPE_MAP = {
    int: 'int64',
    'int64': 'int64',
    float: 'float32',
    'float32': 'float32',
    np.float32: 'float32',
    'uint8': 'uint8',
    np.uint8: 'uint8',
    np.uint64: 'uint64',
    'uint64': 'uint64',
}


def canonicalize_dtype_for_ts(dtype: Any) -> str:
  """Convert from an input datatype to the string version for TensorStore.

  Args:
    dtype: Input datatype (e.g. int, np.float32, etc)

  Returns:
    Tensorstore-compatible dtype string.

  Raises:
    ValueError if input dtype is not known.
  """
  if isinstance(dtype, np.dtype):
    dtype = dtype.name
  elif isinstance(dtype, ts.dtype):
    dtype = dtype.numpy_dtype.name
  if dtype not in _TS_DTYPE_MAP:
    raise ValueError(
        f'Unknown dtype: {dtype}. If you think this should be supported, '
        f'please add conversion from {dtype} to a known TensorStore type in '
        '_TS_DTYPE_MAP')
  return _TS_DTYPE_MAP[dtype]

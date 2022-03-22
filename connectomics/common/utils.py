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

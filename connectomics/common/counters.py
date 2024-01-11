# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Provides counters for monitoring processing."""

import contextlib
import threading
import time
from typing import Iterable


class ThreadsafeCounter:
  """A simple counter that synchronizes all access and provides reset method."""

  def __init__(self):
    self._value = 0
    self._lock = threading.Lock()

  @property
  def value(self):
    with self._lock:
      return self._value

  def inc(self, inc_value: int = 1):
    """Increments the counter value."""
    with self._lock:
      self._value += inc_value

  def reset(self) -> int:
    """Resets the counter and returns the old value."""
    with self._lock:
      old_value = self._value
      self._value = 0
      return old_value


class ThreadsafeCounterStore:
  """A synchronized store for accessing ThreadsafeCounters."""

  def __init__(self):
    self._counters = {}
    self._lock = threading.Lock()

  def get_counter(self, name: str) -> ThreadsafeCounter:
    """Gets counter for tracking processing under the given name."""
    counter = self._counters.get(name)
    if counter is not None:
      return counter

    with self._lock:
      counter = self._counters.get(name)
      if counter is None:
        counter = ThreadsafeCounter()
        self._counters[name] = counter
      return counter

  @contextlib.contextmanager
  def timer_counter(self, name: str):
    """Counts execution time in ms."""
    start = time.time()
    yield
    self.get_counter(name + '-ms').inc(int((time.time() - start) * 1e3))

  def get_nonzero(self) -> Iterable[tuple[str, ThreadsafeCounter]]:
    """Yields name, counter tuples for any counters with value > 0."""
    with self._lock:
      for name, counter in self._counters.items():
        if counter.value > 0:
          yield name, counter

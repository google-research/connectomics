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
"""Beam utilities."""
import contextlib
import time
from typing import TypeVar

import apache_beam as beam

C = TypeVar('C')  # For PCollections


@contextlib.contextmanager
def timer_counter(namespace, name):
  """Counts execution time in ms."""
  start = time.time()
  yield
  beam.metrics.Metrics.counter(namespace, '%s-ms' % name).inc(
      int((time.time() - start) * 1e3))


def counter(namespace, name):
  """Returns a counter."""
  return beam.metrics.Metrics.counter(namespace, name)


class MustFollow(beam.PTransform):
  """Pass-through input, but enforces deferred processing."""

  def __init__(self, must_follow):
    super(MustFollow, self).__init__()
    self._must_follow = must_follow

  def expand(self, pcollection: C) -> C:
    if not self._must_follow:
      return pcollection

    must_follow = self._must_follow | 'empty' >> beam.FlatMap(lambda _: [])
    return pcollection | 'waiting' >> beam.Map(
        lambda x, _: x, beam.pvalue.AsSingleton(must_follow))

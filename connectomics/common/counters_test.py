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
"""Tests for counters.py."""

from absl.testing import absltest
from connectomics.common import counters


class CountersTest(absltest.TestCase):

  def test_threadsafe_counter_store(self):
    store = counters.ThreadsafeCounterStore()
    self.assertEmpty(tuple(store.get_nonzero()))

    counter = store.get_counter('counter')
    self.assertEmpty(tuple(store.get_nonzero()))

    counter.inc()
    nonzero = tuple(store.get_nonzero())
    self.assertLen(nonzero, 1)
    name, counter = nonzero[0]
    self.assertEqual(name, 'counter')
    self.assertEqual(counter.value, 1)

    counter.reset()
    self.assertEmpty(tuple(store.get_nonzero()))


if __name__ == '__main__':
  absltest.main()

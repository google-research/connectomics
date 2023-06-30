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
"""Tests for connectomics.common.beam_utils."""
import time

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util as btutil

from connectomics.common import beam_utils


class MustFollowTest(absltest.TestCase):

  def test_must_follow(self):
    MustFollowTest._MULTIPLIER = 0

    def sleep(x):
      time.sleep(x)
      return x

    def set_multiplier(x):
      MustFollowTest._MULTIPLIER = x
      return x

    with beam.Pipeline(beam.runners.direct.DirectRunner()) as p:
      to_follow = (
          p
          | 'multiplier' >> beam.Create([2])
          | beam.Map(sleep)
          | beam.Map(set_multiplier))
      btutil.assert_that(
          to_follow, btutil.equal_to([2]), label='assert_to_follow')

      following = (
          p
          | 'following' >> beam.Create([1, 2, 3])
          | beam_utils.MustFollow(to_follow)
          | beam.Map(lambda x: x * MustFollowTest._MULTIPLIER))
      btutil.assert_that(
          following, btutil.equal_to([2, 4, 6]), label='assert_following')

  def test_must_follow_chain(self):
    MustFollowTest._MULTIPLIER = 0

    def set_multiplier(x):
      MustFollowTest._MULTIPLIER = x
      return x

    def check_multiplier(x):
      assert MustFollowTest._MULTIPLIER < x
      return x

    with beam.Pipeline(beam.runners.direct.DirectRunner()) as p:
      chain = None
      for i in range(1, 5):
        chain = (
            p
            | 'Make{0}'.format(i) >> beam.Create([i])
            | 'FollowPrev{0}'.format(i) >> beam_utils.MustFollow(chain)
            | 'Check{0}'.format(i) >> beam.Map(check_multiplier)
            | 'Set{0}'.format(i) >> beam.Map(set_multiplier))

      btutil.assert_that(chain, btutil.equal_to([4]), label='assert_chain')


if __name__ == '__main__':
  absltest.main()

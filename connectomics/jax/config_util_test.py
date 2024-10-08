# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Tests for config_util."""

from absl.testing import absltest
from absl.testing import parameterized
from connectomics.jax import config_util as cutil


class ConfigUtilTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_parse_arg_works(self, lazy):
    spec = dict(
        res=224,
        lr=0.1,
        runlocal=False,
        schedule='short',
    )

    def check(result, runlocal, schedule, res, lr):
      self.assertEqual(result.runlocal, runlocal)
      self.assertEqual(result.schedule, schedule)
      self.assertEqual(result.res, res)
      self.assertEqual(result.lr, lr)
      self.assertIsInstance(result.runlocal, bool)
      self.assertIsInstance(result.schedule, str)
      self.assertIsInstance(result.res, int)
      self.assertIsInstance(result.lr, float)

    check(cutil.parse_arg(None, lazy=lazy, **spec), False, 'short', 224, 0.1)
    check(cutil.parse_arg('', lazy=lazy, **spec), False, 'short', 224, 0.1)
    check(cutil.parse_arg('runlocal=True', lazy=lazy, **spec), True, 'short',
          224, 0.1)
    check(cutil.parse_arg('runlocal=False', lazy=lazy, **spec), False, 'short',
          224, 0.1)
    check(cutil.parse_arg('runlocal=', lazy=lazy, **spec), False, 'short', 224,
          0.1)
    check(cutil.parse_arg('runlocal', lazy=lazy, **spec), True, 'short', 224,
          0.1)
    check(cutil.parse_arg('res=128', lazy=lazy, **spec), False, 'short', 128,
          0.1)
    check(cutil.parse_arg('128', lazy=lazy, **spec), False, 'short', 128, 0.1)
    check(cutil.parse_arg('schedule=long', lazy=lazy, **spec), False, 'long',
          224, 0.1)
    check(cutil.parse_arg('runlocal,schedule=long,res=128', lazy=lazy, **spec),
          True, 'long', 128, 0.1)

  @parameterized.parameters(
      (None, {}, {}),
      (None, {'res': 224}, {'res': 224}),
      ('640', {'res': 224}, {'res': 640}),
      ('runlocal', {}, {'runlocal': True}),
      ('res=640,lr=0.1,runlocal=false,schedule=long', {},
       {'res': 640, 'lr': 0.1, 'runlocal': False, 'schedule': 'long'}),
      )
  def test_lazy_parse_arg_works(self, arg, spec, expected):
    self.assertEqual(dict(cutil.parse_arg(arg, lazy=True, **spec)), expected)

  def test_sequence_to_string(self):
    seq = ['a', True, 1, 1.0]
    self.assertEqual(cutil.sequence_to_string(seq), 'a,True,1,1.0')

  def test_string_to_sequence(self):
    self.assertEqual(
        cutil.string_to_sequence('a,True,1,1.0'), ['a', True, 1, 1.0])

if __name__ == '__main__':
  absltest.main()

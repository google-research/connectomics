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
"""Tests for parameter_swap_util."""


from absl.testing import absltest

from connectomics.jax import parameter_replacement_util


class ParameterSwapUtilTest(absltest.TestCase):

  def test_replacement_works_no_replacement(self):
    replacements = {
        "0:key1": {
            "0:key2": "bbb",
            "1:key3": "bbb"
        },
        "1:key1": "bbb",
    }
    parameters = {"0:key1": {"0:key2": "aaa", "1:key3": "aaa"}, "1:key1": "aaa"}
    self.assertEqual(
        3, parameter_replacement_util.get_num_atomic_blocks(parameters))
    self.assertEqual(
        0,
        parameter_replacement_util.replace_final_parameters(
            parameters, replacements, 0))
    self.assertEqual(
        {
            "0:key1": {
                "0:key2": "aaa",
                "1:key3": "aaa"
            },
            "1:key1": "aaa"
        }, parameters)

  def test_replacement_works_some_replacement(self):
    replacements = {
        "0:key1": {
            "0:key2": "bbb",
            "1:key3": "bbb"
        },
        "1:key1": "bbb",
    }
    parameters = {"0:key1": {"0:key2": "aaa", "1:key3": "aaa"}, "1:key1": "aaa"}
    self.assertEqual(
        2,
        parameter_replacement_util.replace_final_parameters(
            parameters, replacements, 2))
    self.assertEqual(
        {
            "0:key1": {
                "0:key2": "aaa",
                "1:key3": "bbb"
            },
            "1:key1": "bbb"
        }, parameters)

  def test_replacement_works_too_many_replacements(self):
    replacements = {
        "0:key1": {
            "0:key2": "bbb",
            "1:key3": "bbb"
        },
        "1:key1": "bbb",
    }
    parameters = {"0:key1": {"0:key2": "aaa", "1:key3": "aaa"}, "1:key1": "aaa"}
    self.assertEqual(
        3,
        parameter_replacement_util.replace_final_parameters(
            parameters, replacements, 12))
    self.assertEqual(
        {
            "0:key1": {
                "0:key2": "bbb",
                "1:key3": "bbb"
            },
            "1:key1": "bbb"
        }, parameters)


if __name__ == "__main__":
  absltest.main()

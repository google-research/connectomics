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
"""Tests for normalization."""

from absl.testing import absltest, parameterized  # pylint: disable=g-multiple-import

from connectomics.jax.models import normalization
from flax import linen as nn

import jax
from jax import random

import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class NormalizationTest(parameterized.TestCase):

  def test_reversible_instance_norm(self):
    e = 1e-5

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        norm = normalization.ReversibleInstanceNorm(
            name='norm',
            use_bias=False,
            use_scale=False,
            epsilon=e,
        )
        x_norm, stats = norm(x)
        y, _ = norm(x_norm, stats)
        return y, x_norm, stats

    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 5, 4, 4, 32))
    (y, x_norm, stats), _ = Foo().init_with_output(key2, x)

    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)
    np.testing.assert_allclose(y, x, atol=1e-6)

    self.assertEqual(x.dtype, x_norm.dtype)
    self.assertEqual(x.shape, x_norm.shape)
    x_gr = x.reshape([2, 5, 4, 4, 32, 1])
    x_norm_test = (
        x_gr - x_gr.mean(axis=[1, 2, 3, 5], keepdims=True)
    ) * jax.lax.rsqrt(x_gr.var(axis=[1, 2, 3, 5], keepdims=True) + e)
    x_norm_test = x_norm_test.reshape([2, 5, 4, 4, 32])
    np.testing.assert_allclose(x_norm_test, x_norm, atol=1e-4)

    self.assertEqual(stats['mean'].shape, (2, 32))
    self.assertEqual(stats['var'].shape, (2, 32))


if __name__ == '__main__':
  absltest.main()

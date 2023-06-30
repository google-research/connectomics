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
"""Tests for jax_utils."""

from absl.testing import absltest
from connectomics.common import jax_utils
import jax
import jax.numpy as jnp
import numpy as np


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._key = jax.random.PRNGKey(seed=42)
    self._shape = (2, 4, 6, 5)
    self._array = jax.random.randint(
        self._key, shape=self._shape, minval=0, maxval=255, dtype=jnp.uint8)

  def _reference_correlation(self, array, loc, axis):
    """Calculates cross-correlation for validation purposes."""
    if axis == 0:
      out_ref = jnp.empty((array.shape[1:]))
      in1 = array
      in2 = array[:, loc[0], loc[1], loc[2]].reshape(-1, 1, 1, 1)
      for b in range(array.shape[1]):
        for c in range(array.shape[2]):
          for d in range(array.shape[3]):
            out_ref = out_ref.at[b, c, d].set(
                jnp.corrcoef(in1[:, b, c, d].reshape(-1),
                             in2.reshape(-1))[0, 1])
    elif axis == -1:
      out_ref = jnp.empty((array.shape[:3]))
      in1 = array
      in2 = array[loc[0], loc[1], loc[2], :].reshape(1, 1, 1, -1)
      for a in range(array.shape[0]):
        for b in range(array.shape[1]):
          for c in range(array.shape[2]):
            out_ref = out_ref.at[a, b, c].set(
                jnp.corrcoef(in1[a, b, c, :].reshape(-1),
                             in2.reshape(-1))[0, 1])
    else:
      raise NotImplementedError
    return out_ref

  def test_correlate_location(self, loc=(1, 2, 3)):
    for axis in (0, -1):
      out_ref = self._reference_correlation(self._array, loc=loc, axis=axis)
      out_scipy = jax_utils.correlate_location(
          self._array, loc=loc, axis=axis, method='scipy')
      out_einsum = jax_utils.correlate_location(
          self._array, loc=loc, axis=axis, method='einsum')
      np.testing.assert_allclose(out_ref, out_scipy, rtol=1e-5)
      np.testing.assert_allclose(out_ref, out_einsum, rtol=1e-5)

  def test_correlate_center(self):
    for axis in (0, -1):
      loc = [l // 2 for l in self._shape[1:]
            ] if axis == 0 else [l // 2 for l in self._shape[:3]]
      out_ref = self._reference_correlation(self._array, loc=loc, axis=axis)
      out_scipy = jax_utils.correlate_center(
          self._array, axis=axis, method='scipy')
      out_einsum = jax_utils.correlate_center(
          self._array, axis=axis, method='einsum')
      np.testing.assert_allclose(out_ref, out_scipy, rtol=1e-5)
      np.testing.assert_allclose(out_ref, out_einsum, rtol=1e-5)

  def test_distances_location(self, loc=(1, 1, 1, 1)):
    out_ref = np.linalg.norm(
        np.moveaxis(np.indices(self._array.shape), 0, -1) - np.array(loc),
        axis=-1)
    out = jax_utils.distances_location(self._array, loc=loc)
    np.testing.assert_almost_equal(
        out_ref.astype(np.float32), np.array(out, dtype=np.float32))

if __name__ == '__main__':
  absltest.main()

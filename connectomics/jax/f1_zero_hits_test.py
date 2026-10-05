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
"""F1 regressions for zero true positives and undefined precision/recall."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized
from clu import metrics as clu_metrics
from connectomics.jax import metrics
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import metrics as sklearn_metrics


class F1ZeroHitsTest(parameterized.TestCase):

  @parameterized.product(
      case=[
          ([True, False], [False, True]),
          ([False, False], [True, False]),
          ([True, False], [False, False]),
      ],
      zero_division=[0.0, 1.0, np.nan],
  )
  def test_incorrect_predictions_have_zero_f1(self, case, zero_division):
    predicted, true = (jnp.array([values]) for values in case)
    p, r, f1 = metrics.precision_recall_f1_bool(predicted, true, zero_division)
    np.testing.assert_array_equal(f1, [0.0])
    np.testing.assert_array_equal(
        metrics.f1_bool(predicted, true, zero_division), f1
    )
    np.testing.assert_allclose(
        p,
        metrics.precision_bool(predicted, true, zero_division),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        r, metrics.recall_bool(predicted, true, zero_division), equal_nan=True
    )

  @parameterized.parameters(0.0, 1.0, np.nan)
  def test_empty_positive_class_uses_the_requested_fallback(
      self, zero_division
  ):
    values = jnp.zeros((2, 2, 3), dtype=bool)
    actual = metrics.f1_bool(values, values, zero_division)
    np.testing.assert_allclose(
        actual, [zero_division, zero_division], equal_nan=True
    )

  @parameterized.parameters(0.0, 1.0, np.nan)
  def test_all_small_binary_cases_match_counts_and_sklearn(self, zero_division):
    vectors = list(itertools.product([False, True], repeat=3))
    pairs = list(itertools.product(vectors, repeat=2))
    predicted = jnp.asarray([p for p, t in pairs])
    true = jnp.asarray([t for p, t in pairs])
    expected = []
    for p, t in pairs:
      tp = sum(a and b for a, b in zip(p, t))
      denominator = sum(p) + sum(t)
      expected.append(2 * tp / denominator if denominator else zero_division)
    sklearn_result = [
        sklearn_metrics.f1_score(t, p, zero_division=zero_division)
        for p, t in pairs
    ]
    np.testing.assert_allclose(expected, sklearn_result, equal_nan=True)
    for fn in (
        lambda p, t: metrics.f1_bool(p, t, zero_division),
        jax.jit(lambda p, t: metrics.f1_bool(p, t, zero_division)),
    ):
      np.testing.assert_allclose(
          fn(predicted, true), expected, rtol=1e-6, equal_nan=True
      )
      np.testing.assert_allclose(
          fn(predicted.reshape(64, 1, 3), true.reshape(64, 1, 3)),
          expected,
          rtol=1e-6,
          equal_nan=True,
      )

  def test_collected_average_includes_defined_zero_scores(self):
    metric = clu_metrics.Average.from_fun(metrics.f1_bool)
    state = metric.from_model_output(
        predictions=jnp.array([[True, False], [True, False]]),
        targets=jnp.array([[False, True], [True, False]]),
    )
    np.testing.assert_allclose(state.compute(), 0.5)

  def test_default_only_treats_no_positive_samples_as_undefined(self):
    predicted = jnp.array([[False, False], [False, True], [True, False]])
    true = jnp.array([[False, False], [True, False], [True, False]])
    np.testing.assert_allclose(
        metrics.f1_bool(predicted, true), [np.nan, 0.0, 1.0], equal_nan=True
    )


if __name__ == '__main__':
  absltest.main()

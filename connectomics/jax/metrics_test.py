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
"""Tests for metrics.

Depending on the metric, tested against dm_pix, sklearn, or manual results.
dm_pix is limited to [batch, x, y, channel].
"""

from absl.testing import absltest
import chex
from connectomics.jax import metrics
import dm_pix
import jax.numpy as jnp
import numpy as np
import scipy.special
import sklearn.metrics


class MetricsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(42)

    # (batch, z, y, x, channel)
    self.vol1 = rng.uniform(0, 1, (4, 28, 32, 36, 2))
    self.vol2 = rng.uniform(0, 1, (4, 28, 32, 36, 2))

    # (batch, x)
    self.boolean_true = np.array([[0, 1, 1, 0, 0, 0, 1, 1, 1, 1]]).astype(bool)
    self.boolean_pred = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0]]).astype(bool)
    tp, fp, fn, tn = 4, 1, 2, 3
    self.boolean_confusion_matrix = [[tp, fp], [fn, tn]]
    self.boolean_precision = tp / (tp + fp)
    self.boolean_recall = tp / (tp + fn)
    self.boolean_f1 = (
        2
        * (self.boolean_precision * self.boolean_recall)
        / (self.boolean_precision + self.boolean_recall)
    )

  def test_mae_integration_against_pix_2d(self):
    mae_pix = dm_pix.mae(self.vol1[:, 0], self.vol2[:, 0])
    mae_neuro = metrics.mae(self.vol1[:, 0], self.vol2[:, 0])
    np.testing.assert_allclose(mae_pix, mae_neuro, atol=1e-6, rtol=1e-6)

  def test_mse_integration_against_pix_2d(self):
    mse_pix = dm_pix.mse(self.vol1[:, 0], self.vol2[:, 0])
    mse_neuro = metrics.mse(self.vol1[:, 0], self.vol2[:, 0])
    np.testing.assert_allclose(mse_pix, mse_neuro, atol=1e-6, rtol=1e-6)

  def test_mape_integration_against_sklearn(self):
    mape_sklearn = sklearn.metrics.mean_absolute_percentage_error(
        y_pred=self.vol1.reshape(-1, 1), y_true=self.vol2.reshape(-1, 1)
    )
    mape_neuro = metrics.mape(
        self.vol1.reshape(1, -1), self.vol2.reshape(1, -1)
    )
    np.testing.assert_allclose(mape_sklearn, mape_neuro, atol=1e-6, rtol=1e-6)

  def test_confusion_matrix_bool_against_manual_result(self):
    cm = metrics.confusion_matrix_bool(self.boolean_pred, self.boolean_true)
    np.testing.assert_array_equal(
        jnp.array([self.boolean_confusion_matrix]), cm
    )

  def test_confusion_matrix_sklearn_against_manual_result(self):
    cm = metrics.confusion_matrix_sklearn(
        ~self.boolean_pred, ~self.boolean_true, labels=[False, True]
    ).transpose((0, 2, 1))
    np.testing.assert_array_equal(
        jnp.array([self.boolean_confusion_matrix]), cm
    )

  def test_precision_bool_against_manual_result(self):
    precision_neuro = metrics.precision_bool(
        self.boolean_pred, self.boolean_true
    )
    np.testing.assert_array_equal(
        jnp.array([self.boolean_precision]), precision_neuro
    )

  def test_precision_bool_against_sklearn(self):
    precision_ref = metrics.make_metric_with_threshold(
        metrics.precision_sklearn, 0.5
    )(self.vol1, self.vol2, zero_division=0.0)
    precision_neuro = metrics.make_metric_with_threshold(
        metrics.precision_bool, 0.5
    )(self.vol1, self.vol2, zero_division=0.0)
    np.testing.assert_allclose(
        precision_ref, precision_neuro, atol=1e-6, rtol=1e-6
    )

  def test_recall_bool_against_manual_result(self):
    recall_neuro = metrics.recall_bool(self.boolean_pred, self.boolean_true)
    np.testing.assert_array_equal(
        jnp.array([self.boolean_recall]), recall_neuro
    )

  def test_recall_bool_against_sklearn(self):
    recall_ref = metrics.make_metric_with_threshold(
        metrics.recall_sklearn, 0.5
    )(self.vol1, self.vol2, zero_division=0.0)
    recall_neuro = metrics.make_metric_with_threshold(metrics.recall_bool, 0.5)(
        self.vol1, self.vol2, zero_division=0.0
    )
    np.testing.assert_allclose(recall_ref, recall_neuro, atol=1e-6, rtol=1e-6)

  def test_f1_bool_against_manual_result(self):
    f1_neuro = metrics.f1_bool(self.boolean_pred, self.boolean_true)
    np.testing.assert_array_equal(jnp.array([self.boolean_f1]), f1_neuro)

  def test_f1_bool_against_sklearn(self):
    f1_ref = metrics.make_metric_with_threshold(metrics.f1_sklearn, 0.5)(
        self.vol1, self.vol2, zero_division=0.0
    )
    f1_neuro = metrics.make_metric_with_threshold(metrics.f1_bool, 0.5)(
        self.vol1, self.vol2, zero_division=0.0
    )
    np.testing.assert_allclose(f1_ref, f1_neuro, atol=1e-6, rtol=1e-6)

  def test_valid_prediction_time(self):
    targets = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    predictions = np.array(
        [[1.0, 1.0, 1.0, 10.0, 10.0], [1.0, 1.0, 1.0, 10.0, 10.0]]
    )
    vpt = metrics.create_vpt_metric(metric_fn=metrics.mse, threshold=0.5)
    metric = vpt.from_model_output(predictions=predictions, targets=targets)
    np.testing.assert_equal(metric.compute(), np.array(3, dtype='int'))

  def test_classification_metrics_binary(self):
    cls = metrics.create_classification_metrics(('neuron', 'glia'))
    m = cls.from_model_output(
        logits=np.array([[-1, 1], [-1, 1], [1, -1]]), labels=np.array([0, 1, 1])
    )
    actual = m.compute()

    self.assertEqual(actual['precision__glia'], 0.5)
    self.assertEqual(actual['precision__neuron'], 0)
    self.assertEqual(actual['recall__neuron'], 0)
    self.assertEqual(actual['recall__glia'], 0.5)
    self.assertEqual(actual['f1__neuron'], 0)
    self.assertEqual(actual['f1__glia'], 0.5)

  def test_classification_metrics_multiclass(self):
    cls = metrics.create_classification_metrics(('axon', 'dend', 'glia'))
    l1, l2, l3 = (
        scipy.special.logit(0.1),
        scipy.special.logit(0.2),
        scipy.special.logit(0.7),
    )

    m = cls.from_model_output(
        logits=np.array([[l1, l2, l3], [l2, l3, l1], [l2, l1, l3]]),  # gdg
        labels=np.array([0, 1, 2]),  # adg
    )
    actual = m.compute()

    self.assertEqual(actual['precision__axon'], 0)
    self.assertEqual(actual['recall__axon'], 0)
    self.assertEqual(actual['f1__axon'], 0)

    self.assertEqual(actual['precision__glia'], 0.5)
    self.assertEqual(actual['recall__glia'], 1.)
    self.assertEqual(actual['f1__glia'], 2/3.)

    self.assertEqual(actual['precision__dend'], 1)
    self.assertEqual(actual['recall__dend'], 1)
    self.assertEqual(actual['f1__dend'], 1)

  def test_count(self):
    count = metrics.Count.from_fun(metrics.nonzero_weight)
    actual = count.from_model_output(
        weight=jnp.asarray([0.5, 0.0, 0.25]), mask=None
    ).compute()
    chex.assert_trees_all_close(actual, 2)


if __name__ == '__main__':
  absltest.main()

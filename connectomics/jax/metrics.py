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
"""Metrics.

All metrics assume a leading batch dimension that is preserved and assume
inputs `predictions`, `targets`. All functions are compatible with clu.metrics
constructed from functions. Constructed relative metrics additionally require
the keyword argument `baseline`.
"""

from collections.abc import Callable
from typing import Any, Sequence

from clu import metric_writers
from clu import metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
import sklearn.metrics

Array = metric_writers.interface.Array
Scalar = metric_writers.interface.Scalar


def get_metrics_collection_from_dict(
    metrics_dict: dict[str, Any], prefix: str = ''
) -> type[metrics.Collection]:
  """Gets metrics collection from dict with optional prefix."""
  return metrics.Collection.create(
      **{f'{prefix}{k}': v for k, v in metrics_dict.items()}
  )


def make_dict_of_scalars(
    metrics_dict: dict[str, Scalar | Array],
    prefix_keys: str = '',
    prefix_vector: str = '/',
) -> dict[str, Scalar]:
  """Converts vectors to scalars in metrics dict."""
  metrics_dict_compat = dict()
  for k, v in metrics_dict.items():
    if isinstance(v, int) or isinstance(v, float):
      metrics_dict_compat[f'{prefix_keys}{k}'] = v
    elif isinstance(v, np.ndarray) or isinstance(v, jnp.ndarray):
      if v.ndim == 0:
        metrics_dict_compat[f'{prefix_keys}{k}'] = v
      elif v.ndim == 1:
        for i, v_el in enumerate(v):
          metrics_dict_compat[f'{prefix_keys}{k}{prefix_vector}{i+1}'] = v_el
      else:
        raise ValueError('Only scalars or vectors are allowed.')
    else:
      raise ValueError('Unsupported type.')
  return metrics_dict_compat


def make_relative_metric(
    metric: Callable[..., jnp.ndarray],
) -> Callable[..., jnp.ndarray]:
  """Construct relative metric to a baseline given base metric."""

  def _relative_metric(
      predictions: jnp.ndarray,
      targets: jnp.ndarray,
      baseline: jnp.ndarray,
      **kwargs,
  ) -> jnp.ndarray:
    metric_model = metric(predictions=predictions, targets=targets, **kwargs)
    metric_baseline = metric(predictions=baseline, targets=targets, **kwargs)
    return metric_model / metric_baseline

  return _relative_metric


def make_per_step_metric(
    metric: Callable[..., jnp.ndarray],
) -> Callable[..., jnp.ndarray]:
  """Construct per-step metric."""

  def _per_step_metric(
      predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs
  ) -> jnp.ndarray:
    assert predictions.shape == targets.shape
    assert len(targets.shape) >= 2
    kwargs['video'] = False  # Only needed for video_forecasting.metrics.ssim
    batch, timesteps = targets.shape[:2]
    predictions = predictions.reshape(batch * timesteps, *targets.shape[2:])
    targets = targets.reshape(batch * timesteps, *targets.shape[2:])
    score = metric(predictions=predictions, targets=targets, **kwargs)
    return score.reshape(batch, timesteps)

  return _per_step_metric


@flax.struct.dataclass
class PerStepAverage(metrics.Metric):
  """Average metric with additional kept leading dimension (e.g. steps).

  Assumes inputs of shape of shape (batch, steps) and averages to (steps,).
  """

  total: jnp.ndarray
  count: jnp.ndarray

  @classmethod
  def empty(cls) -> Any:
    return cls(total=jnp.array(0, jnp.float32), count=jnp.array(0, jnp.int32))

  @classmethod
  def from_model_output(cls, values: jnp.ndarray, mask: Any = None, **_) -> Any:
    assert values.ndim >= 2, 'Vector Average requires per sample steps'
    assert mask is None, 'Mask not supported'
    batch, timesteps = values.shape[:2]
    total = values.reshape(batch, timesteps, -1).sum(axis=(0, 2))
    return cls(total=total, count=batch)  # pytype: disable=wrong-arg-types  # jnp-array

  def merge(self, other: Any) -> Any:
    return type(self)(
        total=self.total + other.total,
        count=self.count + other.count,
    )

  def compute(self) -> Any:
    return self.total / self.count


def make_metric_with_threshold(
    metric: Callable[..., jnp.ndarray], threshold: float
) -> Callable[..., jnp.ndarray]:
  """Construct metric that is applied after thresholding to boolean array."""

  def _metric_with_threshold(
      predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs
  ) -> jnp.ndarray:
    return metric(
        predictions=predictions > threshold,
        targets=targets > threshold,
        **kwargs,
    )

  return _metric_with_threshold


def mse(predictions: jnp.ndarray, targets: jnp.ndarray, **_) -> jnp.ndarray:
  """Compute mean squared error per example."""
  assert predictions.shape == targets.shape
  axes = tuple(range(1, targets.ndim))
  return jnp.mean(jnp.square(targets - predictions), axis=axes)


def mae(predictions: jnp.ndarray, targets: jnp.ndarray, **_) -> jnp.ndarray:
  """Compute mean absolute error per example."""
  assert predictions.shape == targets.shape
  axes = tuple(range(1, targets.ndim))
  return jnp.mean(jnp.abs(targets - predictions), axis=axes)


def mape(predictions: jnp.ndarray, targets: jnp.ndarray, **_) -> jnp.ndarray:
  """Compute mean absolute percentage error per example."""
  assert predictions.shape == targets.shape
  eps = jnp.finfo(targets.dtype).eps
  axes = tuple(range(1, targets.ndim))
  return jnp.mean(
      jnp.abs(predictions - targets) / jnp.maximum(jnp.abs(targets), eps),
      axis=axes,
  )


@jax.jit
def _confusion_matrix_bool_1d(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, **_
) -> jnp.ndarray:
  """Calculates confusion matrix for boolean 1d-arrays."""
  return jnp.bincount(2 * y_true + y_pred, minlength=4, length=4).reshape(2, 2)


def confusion_matrix_bool(
    predictions: jnp.ndarray, targets: jnp.ndarray, **_
) -> jnp.ndarray:
  """Calculates confusion matrix for boolean arrays.

  Args:
    predictions: Array of boolean predictions with leading batch dimension.
    targets: Array of boolean targets with leading batch dimension.

  Returns:
    Confusion matrix, with True values as positive class, laid out as follows:
      tp fp
      fn tn,
    where tp = true pos., fp = false pos., fn = false neg., and tn = true neg.
  """
  assert predictions.dtype == targets.dtype == bool
  assert predictions.shape == targets.shape
  shape = (targets.shape[0], -1)

  predictions = predictions.reshape(*shape)
  targets = targets.reshape(*shape)

  cm_batched = jax.vmap(_confusion_matrix_bool_1d, 0, 0)
  return cm_batched(~targets, ~predictions).transpose((0, 2, 1))


def confusion_matrix_sklearn(
    predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
  """Calculates confusion matrix with sklearn.

  In the case of boolean arrays, the implementation in `confusion_matrix_bool`
  can be significantly faster, see also [1].

  To match the return format of `confusion_matrix_bool` for boolean arrays, use:
    confusion_matrix_sklearn(
      ~predictions, ~targets, labels=[False, True]).transpose((0, 2, 1))

  Args:
    predictions: Array of boolean predictions with leading batch dimension.
    targets: Array of boolean targets with leading batch dimension.
    **kwargs: Passed to sklearn.metrics.confusion_matrix.

  Returns:
    Confusion matrix with leading batch dimension whose i-th row and j-th
    column entry indicates the number of samples with true label being i-th
    class and predicted label being j-th class.

  References:
    [1]: https://github.com/scikit-learn/scikit-learn/issues/15388
  """
  assert predictions.shape == targets.shape
  shape = (targets.shape[0], -1)

  predictions = predictions.reshape(*shape)
  targets = targets.reshape(*shape)

  res = []
  for batch in range(shape[0]):
    res.append(
        sklearn.metrics.confusion_matrix(
            y_true=targets[batch, :], y_pred=predictions[batch, :], **kwargs
        )
    )
  return jnp.array(res)


def precision_bool(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    zero_division: float = jnp.nan,
    **_,
) -> jnp.ndarray:
  """Compute precision for boolean arrays."""
  assert predictions.dtype == targets.dtype == bool
  assert predictions.shape == targets.shape
  cm = confusion_matrix_bool(predictions=predictions, targets=targets)
  # precision: tp / (tp + fp)
  numerator = cm[:, 0, 0]
  denominator = cm[:, 0, 0] + cm[:, 0, 1]
  return jnp.where(denominator > 0, numerator / denominator, zero_division)


def precision_sklearn(
    predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
  """Compute precision with sklearn."""
  assert predictions.shape == targets.shape
  shape = (targets.shape[0], -1)
  return jnp.array([
      sklearn.metrics.precision_score(
          y_true=targets.reshape(*shape)[b, :],
          y_pred=predictions.reshape(*shape)[b, :],
          **kwargs,
      )
      for b in range(shape[0])
  ])


def recall_bool(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    zero_division: float = jnp.nan,
    **_,
) -> jnp.ndarray:
  """Compute recall for boolean arrays."""
  assert predictions.dtype == targets.dtype == bool
  assert predictions.shape == targets.shape
  cm = confusion_matrix_bool(predictions=predictions, targets=targets)
  # recall: tp / (tp + fn)
  numerator = cm[:, 0, 0]
  denominator = cm[:, 0, 0] + cm[:, 1, 0]
  return jnp.where(denominator > 0, numerator / denominator, zero_division)


def recall_sklearn(
    predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
  """Compute recall with sklearn."""
  assert predictions.shape == targets.shape
  shape = (targets.shape[0], -1)
  return jnp.array([
      sklearn.metrics.recall_score(
          y_true=targets.reshape(*shape)[b, :],
          y_pred=predictions.reshape(*shape)[b, :],
          **kwargs,
      )
      for b in range(shape[0])
  ])


def precision_recall_f1_bool(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    zero_division: float = jnp.nan,
    **_,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute precision, recall, f1 score for boolean arrays."""
  assert predictions.dtype == targets.dtype == bool
  assert predictions.shape == targets.shape
  cm = confusion_matrix_bool(predictions=predictions, targets=targets)

  # precision: tp / (tp + fp)
  numerator = cm[:, 0, 0]
  denominator = cm[:, 0, 0] + cm[:, 0, 1]
  p = jnp.where(denominator > 0, numerator / denominator, zero_division)

  # recall: tp / (tp + fn)
  denominator = cm[:, 0, 0] + cm[:, 1, 0]
  r = jnp.where(denominator > 0, numerator / denominator, zero_division)

  # f1: 2 * (precision * recall) / (precision + recall)
  numerator = 2 * p * r
  denominator = p + r
  f1 = jnp.where(denominator > 0, numerator / denominator, zero_division)

  return p, r, f1


def f1_bool(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    zero_division: float = jnp.nan,
    **_,
) -> jnp.ndarray:
  """Compute f1 score for boolean arrays."""
  _, _, f1 = precision_recall_f1_bool(predictions, targets, zero_division)
  return f1


def f1_sklearn(
    predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
  """Compute f1 score with sklearn."""
  assert predictions.shape == targets.shape
  shape = (targets.shape[0], -1)
  return jnp.array([
      sklearn.metrics.f1_score(
          y_true=targets.reshape(*shape)[b, :],
          y_pred=predictions.reshape(*shape)[b, :],
          **kwargs,
      )
      for b in range(shape[0])
  ])


def create_vpt_metric(metric_fn: Any, threshold: float) -> type[metrics.Metric]:
  """Creates metric to compute valid prediction time (VPT).

  Assumes inputs of shape (batch, steps) and returns VPT as an integer,
  computing argmin over steps for `metric_fn(predictions, targets) > threshold`.

  Args:
    metric_fn: Metric function.
    threshold: Threshold.

  Returns:
    VPT metric.
  """

  @flax.struct.dataclass
  class _ValidPredictionTime(PerStepAverage):
    """Valid Prediction Time metric."""

    def compute(self) -> Any:
      return jnp.min(jnp.argwhere(self.total / self.count > threshold))

  return _ValidPredictionTime.from_fun(make_per_step_metric(metric_fn))


def nonzero_weight(weight: jnp.ndarray, **_):
  return (weight > 0).astype(jnp.int64)


@flax.struct.dataclass
class Count(metrics.Metric):
  """Counts positive values in the input."""

  count: jnp.ndarray

  @classmethod
  def from_model_output(
      cls, inputs: jnp.ndarray, mask: jnp.ndarray | None = None, **_
  ) -> metrics.Metric:
    return cls(count=inputs.sum())

  def merge(self, other: 'Count') -> 'Count':
    return type(self)(count=self.count + other.count)

  def compute(self) -> Any:
    return self.count


def create_classification_metrics(
    class_names: Sequence[str],
) -> type[metrics.CollectingMetric]:
  """Creates classification metrics for N classes."""

  @flax.struct.dataclass
  class ClassificationMetrics(
      metrics.CollectingMetric.from_outputs(('labels', 'logits'))
  ):
    """Computes precision, recall, F1, auc_pr and roc_auc per class.

    Data (labels, logits) is collected on the host, and summarized into
    metrics when compute() is called.
    """

    classes: Sequence[str] = class_names

    def compute(self) -> dict[str, float]:
      """Computes the metrics."""
      values = super().compute()
      labels = np.array(values['labels'])
      logits = np.array(values['logits'])

      labels = labels.ravel()
      logits = logits.reshape([-1, logits.shape[-1]])

      if logits.shape[-1] != len(self.classes):
        raise ValueError(
            f'Number of classes {len(self.classes)} does not match logits'
            f' dimension {logits.shape[-1]}.'
        )

      with jax.default_device(jax.local_devices(backend='cpu')[0]):
        labels_1hot = np.asarray(
            jax.nn.one_hot(labels, num_classes=len(self.classes))
        )

      prob = scipy.special.softmax(logits, axis=-1)
      pred = prob.argmax(axis=-1)

      if prob.shape[-1] == 2:
        roc_prob = prob[:, 0]
      else:
        roc_prob = prob

      precision, recall, f1, _ = (
          sklearn.metrics.precision_recall_fscore_support(labels, pred)
      )
      roc_auc = sklearn.metrics.roc_auc_score(
          labels, roc_prob, multi_class='ovr'
      )
      auc_pr = sklearn.metrics.average_precision_score(
          labels_1hot, prob, average=None
      )

      ret = {
          'roc_auc': roc_auc,
      }

      for i, name in enumerate(self.classes):
        ret[f'precision__{name}'] = precision[i]
        ret[f'recall__{name}'] = recall[i]
        ret[f'f1__{name}'] = f1[i]
        ret[f'auc_pr__{name}'] = auc_pr[i]

      return ret

  return ClassificationMetrics

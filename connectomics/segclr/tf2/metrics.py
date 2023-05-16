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
"""Training utilities."""

from absl import logging
import tensorflow.compat.v2 as tf


def _float_metric_value(metric):
  """Gets the value of a float-value keras metric."""
  return metric.result().numpy().astype(float)


def update_pretrain_metrics_train(
    contrast_loss_metric,
    correlation_loss_metric,
    contrast_acc_metric,
    contrast_entropy_metric,
    contrast_loss,
    corr_loss,
    logits_con,
    labels_con,
):
  """Updated pretraining metrics."""
  contrast_loss_metric.update_state(contrast_loss)
  correlation_loss_metric.update_state(corr_loss)

  contrast_acc_val = tf.equal(
      tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1)
  )
  contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
  contrast_acc_metric.update_state(contrast_acc_val)

  prob_con = tf.nn.softmax(logits_con)
  entropy_con = -tf.reduce_mean(
      tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1)
  )
  contrast_entropy_metric.update_state(entropy_con)


def log_and_write_metrics_to_summary(all_metrics, global_step):
  for metric in all_metrics:
    metric_value = _float_metric_value(metric)
    logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
    tf.summary.scalar(metric.name, metric_value, step=global_step)

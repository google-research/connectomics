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
"""Contrastive loss functions.

Based on objective from SimCLR. Added additional calculation in loss function:
The loss function computes mean correlation between embedding dimensions and 
mean entropy which can both be used as additional losses.
"""

import tensorflow.compat.v2 as tf

# Scaling parameter for one hot encoding vectors. Needs to be much larger
# than the expected logits. Value copied from SimCLR codebase.
_LARGE_NUM = 1e9


def add_supervised_loss(labels, logits):
  """Compute mean supervised loss over local batch."""
  losses = tf.keras.losses.CategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE
  )(labels, logits)
  return tf.reduce_mean(losses)


def tf_cov(x_in):
  """Computes the cross correlation matrix of all data dimensions."""
  x = x_in / tf.math.reduce_std(x_in, axis=0)
  mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
  mx = tf.matmul(tf.transpose(mean_x), mean_x)
  vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
  cov_xx = vx - mx
  return cov_xx


def add_contrastive_loss(
    hidden,
    hidden_norm=True,
    temperature=1.0,
    strategy=None,
    calc_correlation=False,
    calc_entropy=False,
    weights=1.0,
):
  """Compute NT-XENT loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    strategy: context information for tpu.
    calc_correlation: Calculates embedding dimension correlations
    calc_entropy: Calculates embedding entropy
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
    The mean correlation of the embedding vectors
    The entropy loss
  """
  # Get (normalized) hidden1 and hidden2. hidden1 and hidden2 are the
  # corresponding projections of the positive examples.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if strategy is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
    hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
    enlarged_batch_size = tf.shape(hidden1_large)[0]
    replica_context = tf.distribute.get_replica_context()
    replica_id = tf.cast(
        tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32
    )
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
    masks = tf.one_hot(labels_idx, enlarged_batch_size)
  else:
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_aa = logits_aa - masks * _LARGE_NUM
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - masks * _LARGE_NUM
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

  loss_a = tf.nn.softmax_cross_entropy_with_logits(
      labels, tf.concat([logits_ab, logits_aa], 1)
  )
  loss_b = tf.nn.softmax_cross_entropy_with_logits(
      labels, tf.concat([logits_ba, logits_bb], 1)
  )
  loss = tf.reduce_mean((loss_a + loss_b) * weights)

  if calc_correlation:
    hidden3_large = tf.concat([hidden1_large, hidden2_large], axis=0)
    emb_cross_coerr = tf_cov(hidden3_large)
    size = tf.cast(tf.shape(emb_cross_coerr)[0], tf.float32)
    correlation_loss = tf.reduce_mean(emb_cross_coerr**2) - (1. / size)
  else:
    correlation_loss = 0

  if calc_entropy:
    probs = tf.nn.softmax(tf.concat([hidden1_large, hidden2_large], axis=0))
    entropy_loss = tf.reduce_mean(
        tf.reduce_mean(probs * tf.math.log(probs + 1e-8), -1)
    )
  else:
    entropy_loss = 0

  return loss, logits_ab, labels, correlation_loss, entropy_loss


def collect_overall_results(hidden, hidden_norm=True, tpu_context=None):
  """Collects overall metrics."""

  # Get (normalized) hidden1 and hidden2.

  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if tpu_context is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, tpu_context)
    hidden2_large = tpu_cross_replica_concat(hidden2, tpu_context)
    enlarged_batch_size = tf.shape(hidden1_large)[0]

    labels = tf.one_hot(tf.range(enlarged_batch_size), enlarged_batch_size)
    logits = tf.matmul(hidden1_large, hidden2_large, transpose_b=True)
  else:
    labels = tf.one_hot(tf.range(batch_size), batch_size)
    logits = tf.matmul(hidden1, hidden2, transpose_b=True)

  probs = tf.nn.softmax(logits)
  prob_entropy = -tf.reduce_mean(
      tf.reduce_sum(probs * tf.math.log(probs + 1e-8), -1)
  )
  logit_entropy = -tf.reduce_mean(
      tf.reduce_sum(logits * tf.math.log(logits + 1e-8), -1)
  )

  topk_acc = {}
  for k in [1, 5]:
    topk_acc[k] = tf.reduce_mean(
        tf.keras.metrics.top_k_categorical_accuracy(labels, logits, k=k)
    )

  sqrt_d = tf.sqrt(tf.cast(hidden.shape[-1], tf.float32))
  std_x_sqrtd = tf.reduce_mean(tf.math.reduce_std(hidden, -1)) * sqrt_d

  return prob_entropy, logit_entropy, topk_acc, std_x_sqrtd


def tpu_cross_replica_concat(tensor, strategy=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if strategy is None or strategy.num_replicas_in_sync <= 1:
    return tensor

  num_replicas = strategy.num_replicas_in_sync

  replica_context = tf.distribute.get_replica_context()
  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[replica_context.replica_id_in_sync_group]],
        updates=[tensor],
        shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0),
    )

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = replica_context.all_reduce(
        tf.distribute.ReduceOp.SUM, ext_tensor
    )

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])

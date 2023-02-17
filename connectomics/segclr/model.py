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
"""Model specification."""

from typing import Any, Optional, Sequence

from connectomics.segclr import model_util
from connectomics.segclr import objective
from connectomics.segclr import resnet
import numpy as np
from simclr import lars_optimizer
import tensorflow.compat.v1 as tf
# pylint: disable=g-deprecated-tf-checker
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v2 as tf2  # For summaries only

DEFAULT_MOMENTUM = 0.99
DEFAULT_EPSILON = 1e-5


def model_gen(model_dir: str,
              n_data_channels: int,
              use_bn: bool = True,
              resnet_kind: str = "resnet18",
              base_learning_rate: float = 0.01,
              num_proj_layers: int = 3,
              num_bottleneck_layers: int = 3,
              proj_out_dim: int = 128,
              bottleneck_dim: int = 128,
              simsiam_df: int = 4,
              temperature: float = 0.1,
              weight_decay: float = 0.9,
              momentum: float = 1e-4,
              batch_norm_decay: float = 0.9,
              train_summary_steps: int = 100,
              use_tpu: bool = False,
              mode: str = "train",
              distance_head: bool = False,
              separate_model_training: bool = False,
              distance_bin_bounds: Optional[Sequence[int]] = None,
              multi_head_stopg: bool = True,
              partial_training: bool = False,
              emb_correlation_reg: float = 0,
              bottle_entropy_loss_mult: float = 0,
              single_projection_head: bool = True,
              max_steps: int = 500000,
              warmup_steps: int = 0,
              loss_name: str = "ntxent") -> Any:
  """Generates model_fn.

  Args:
    model_dir: path to model directory
    n_data_channels:
    use_bn: whether to use batch norm
    resnet_kind: resnet18 and resnet50 are supported
    base_learning_rate: starting learning rate
    num_proj_layers: numnber of layers in projection head
    num_bottleneck_layers:
    proj_out_dim: size of projection head output layer
    bottleneck_dim: bottleneck dimensions
    simsiam_df: reduction factor in simsiam projection head
    temperature: temperature for loss
    weight_decay: weight decay
    momentum: momentum
    batch_norm_decay: batch norm decay
    train_summary_steps: how often to record the summaries
    use_tpu: using TPU?
    mode: operation mode; either 'pred', 'eval' or 'train'
    distance_head: whether to add a distance prediction head during training
    separate_model_training: only relevant when training with multiple encoders
      bin. If True, every encoder has its own projection head.
    distance_bin_bounds: boundaries of the distance bins. The length of this
      list is n_bins+2
    multi_head_stopg: only relevant when training with multiple distance
      bin. If True, the gradient of a single projection head is only passed
      through a distinct subspace of the bottleneck layer
    partial_training: only relevant when training with multiple distance bin. If
      True, a projection head only "sees" its distinct subspace of the
      bottleneck layer
    emb_correlation_reg: Adds a loss that decorrelates the embedding dimensions
    bottle_entropy_loss_mult: Multiplier on the entropy loss
    single_projection_head: only relevant when training with multiple distance
      bin. If False, every distance bin has its own projection head.
    max_steps: maximum steps to take during training
    warmup_steps: number of warmup steps
    loss_name: Name of loss (ntxent, simsiam)

  Returns:
    _model_fn: function to be passed to tensorflow estimator
  """

  def _model_fn(features, labels=None, params=None):
    del labels
    if params is None:
      params = {}

    tpu_context = params["context"] if "context" in params else None

    # Base model
    bottle_outs = []
    proj_outs = []
    bottle_out_dim = 0

    if mode == "train":
      paired_data_mc = tf.split(
          tf.concat(tf.split(features["data"], 2, -1), 0), n_data_channels, -1)
    elif mode in ["eval", "pred"]:
      paired_data_mc = tf.split(features["data"], n_data_channels, -1)
    else:
      raise ValueError("mode must be one of ['train', 'eval', 'pred']")

    for channel_id in range(n_data_channels):
      with tf.variable_scope(f"base_model_{channel_id}"):
        resnet_rep = resnet.resnet_stack(
            resnet_kind,
            paired_data_mc[channel_id],
            batch_norm=use_bn,
            batch_norm_decay=batch_norm_decay,
            is_training=mode == "train")

      if num_bottleneck_layers > 0:
        with tf.variable_scope(f"bottleneck_head_{channel_id}"):
          bottle_outs.append(
              model_util.projection_head(
                  resnet_rep,
                  out_dim=bottleneck_dim,
                  num_layers=num_bottleneck_layers,
                  use_bn=use_bn,
                  batch_norm_decay=batch_norm_decay,
                  is_training=mode == "train"))
        bottle_out_dim = bottleneck_dim
      else:
        bottle_outs.append(resnet_rep)
        bottle_out_dim = 512 * n_data_channels

    if mode in ["eval", "pred"]:
      features["embeddings"] = tf.concat(bottle_outs, axis=1)

      del features["data"]
      # TODO(sdorkenw): remove dependency on estimator.
      return tf_estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=features)

    elif mode == "train":
      if bottle_entropy_loss_mult > 0:
        _, _, _, corr_loss, entropy_loss = objective.add_ntxent_loss(
            tf.concat(bottle_outs, axis=1),
            temperature=temperature,
            hidden_norm=True,
            calc_correlation=False,
            tpu_context=tpu_context)
      else:
        entropy_loss = 0

      # Assign projection heads
      distance_bin_bounds_np = np.array(distance_bin_bounds, dtype=np.int64)
      contrast_losses = []
      correlation_losses = []
      distance_losses = []

      # Number of projection heads
      if single_projection_head:
        n_projection_heads = 1
      else:
        n_projection_heads = len(distance_bin_bounds_np) - 1

      # Max supported size of embeddings due to number of projection heads
      red_emb_dim = (bottle_out_dim // n_projection_heads) * n_projection_heads

      # Keep multi-encoder embeddings separate or combine them
      if separate_model_training:
        partial_embeddings = bottle_outs
      else:
        partial_embeddings = [tf.concat(bottle_outs, axis=1)]

      # Iterate over partial embeddings (usually 1)
      for i_partial, partial_embedding in enumerate(partial_embeddings):
        # Split embeddings up for projection heads
        partial_embedding_split = tf.split(partial_embedding[:, :red_emb_dim],
                                           n_projection_heads, 1)

        # Every projection may see the entire embedding space
        # (partial_training == False). Its loss
        # might only be backpropagated through a part of the embedding
        # (multi_head_stopg == True)
        for i_head in range(n_projection_heads):
          # Calculating mask
          if n_projection_heads > 1:
            lower = tf.greater_equal(features["distance"],
                                     distance_bin_bounds_np[i_head])
            upper = tf.less(features["distance"],
                            distance_bin_bounds_np[i_head + 1])
            proj_loss_mask = tf.reshape(tf.logical_and(lower, upper), [-1])
          else:
            proj_loss_mask = tf.reshape(
                tf.greater_equal(features["distance"], 0), [-1]
            )

          if multi_head_stopg or not partial_training:
            sub_embeddings = []
            for i_stop_head in range(n_projection_heads):
              if i_head != i_stop_head and multi_head_stopg:
                sub_embeddings.append(
                    tf.stop_gradient(partial_embedding_split[i_stop_head]))
              else:
                sub_embeddings.append(partial_embedding_split[i_stop_head])

            head_embedding = tf.concat(sub_embeddings, 1)
          else:
            head_embedding = partial_embedding_split[i_head]

          with tf.variable_scope(f"projection_head_{i_partial}_{i_head}"):
            proj_out = model_util.projection_head(
                head_embedding,
                out_dim=proj_out_dim,
                num_layers=num_proj_layers,
                use_bn=use_bn,
                batch_norm_decay=batch_norm_decay,
                is_training=mode == "train")

          if loss_name == "simsiam":
            with tf.variable_scope(
                f"simsiam_projection_head_{i_partial}_{i_head}"):
              pred_out = model_util.simsiam_projection_head(
                  proj_out,
                  use_bn=use_bn,
                  batch_norm_decay=batch_norm_decay,
                  simsiam_df=simsiam_df,
                  is_training=mode == "train")

              contrast_loss = objective.add_simsiam_loss(
                  proj_out,
                  pred_out,
                  weights=tf.cast(proj_loss_mask, dtype=tf.float32),
              )
              corr_part_loss = 0
          else:
            contrast_loss, _, _, corr_part_loss, _ = objective.add_ntxent_loss(
                proj_out,
                temperature=temperature,
                hidden_norm=True,
                calc_correlation=True,
                weights=tf.cast(proj_loss_mask, dtype=tf.float32),
                tpu_context=tpu_context)

          correlation_losses.append(corr_part_loss)
          contrast_losses.append(contrast_loss)
          proj_outs.append(proj_out)

        if distance_head:
          with tf.variable_scope(f"distance_head_{i_partial}"):
            dist_out = model_util.projection_head(
                tf.concat(tf.split(partial_embedding, 2, 0), axis=-1),
                out_dim=1,
                num_layers=num_proj_layers,
                use_bn=use_bn,
                batch_norm_decay=batch_norm_decay,
                is_training=mode == "train")
            log_dists = tf.log(
                tf.cast(features["distance"], dtype=tf.float32) / 1000.)
            distance_losses.append(
                tf.losses.mean_squared_error(log_dists, dist_out))

      global_step = tf.train.get_or_create_global_step()

      learning_rate_step = model_util.learning_rate_step(
          base_learning_rate, warmup_steps, max_steps)

      corr_loss = tf.reduce_mean(correlation_losses) * emb_correlation_reg

      tf.compat.v1.losses.add_loss(
          corr_loss, loss_collection=tf.GraphKeys.LOSSES)

      tf.compat.v1.losses.add_loss(
          entropy_loss * bottle_entropy_loss_mult,
          loss_collection=tf.GraphKeys.LOSSES)

      loss = tf.losses.get_total_loss()

      # Summaries
      # Compute stats for the summary.
      summary_writer = tf2.summary.create_file_writer(
          model_dir, filename_suffix="train", name="train_summary_writer")

      with tf.control_dependencies([summary_writer.init()]):
        with summary_writer.as_default():
          should_record = tf.math.equal(
              tf.math.floormod(global_step, train_summary_steps), 0)

          with tf2.summary.record_if(should_record):
            tf2.summary.scalar(
                "train/overview/learning_rate",
                learning_rate_step,
                step=global_step)

            for i_bottle, bottle_out in enumerate(bottle_outs):
              (
                  bottle_prob_entropy,
                  bottle_logit_entropy,
                  bottle_topk_acc,
                  bottle_out_std_x_sqrtd,
              ) = objective.collect_overall_results(
                  bottle_out, hidden_norm=True, tpu_context=tpu_context
              )

              tf2.summary.scalar(
                  f"train/std_x_sqrtd/bottleneck_{i_bottle}",
                  bottle_out_std_x_sqrtd,
                  step=global_step)
              tf2.summary.scalar(
                  f"train/contrastive_prob_entropy/bottleneck_{i_bottle}",
                  bottle_prob_entropy,
                  step=global_step)
              tf2.summary.scalar(
                  f"train/contrastive_logit_entropy/bottleneck_{i_bottle}",
                  bottle_logit_entropy,
                  step=global_step)
              for k in bottle_topk_acc:
                tf2.summary.scalar(
                    f"train/contrastive_acc_top{k}/bottleneck_{i_bottle}",
                    bottle_topk_acc[k],
                    step=global_step)

              (
                  proj_prob_entropy,
                  proj_logit_entropy,
                  proj_topk_acc,
                  proj_out_std_x_sqrtd,
              ) = objective.collect_overall_results(
                  proj_outs[i_bottle], hidden_norm=True, tpu_context=tpu_context
              )

              tf2.summary.scalar(
                  f"train/std_x_sqrtd/proj_{i_bottle}",
                  proj_out_std_x_sqrtd,
                  step=global_step)
              tf2.summary.scalar(
                  f"train/contrastive_prob_entropy/proj_{i_bottle}",
                  proj_prob_entropy,
                  step=global_step)
              tf2.summary.scalar(
                  f"train/contrastive_logit_entropy/proj_{i_bottle}",
                  proj_logit_entropy,
                  step=global_step)
              for k in proj_topk_acc:
                tf2.summary.scalar(
                    f"train/contrastive_acc_top{k}/proj_{i_bottle}",
                    proj_topk_acc[k],
                    step=global_step)

            if len(bottle_outs) > 1:
              (
                  bottle_prob_entropy,
                  bottle_logit_entropy,
                  bottle_topk_acc,
                  bottle_out_std_x_sqrtd,
              ) = objective.collect_overall_results(
                  tf.concat(bottle_outs, axis=1),
                  hidden_norm=True,
                  tpu_context=tpu_context,
              )

              tf2.summary.scalar(
                  "train/std_x_sqrtd/bottleneck_concat",
                  bottle_out_std_x_sqrtd,
                  step=global_step)
              tf2.summary.scalar(
                  "train/contrastive_prob_entropy/bottleneck_concat",
                  bottle_prob_entropy,
                  step=global_step)
              tf2.summary.scalar(
                  "train/contrastive_logit_entropy/bottleneck_concat",
                  bottle_logit_entropy,
                  step=global_step)
              for k in bottle_topk_acc:
                tf2.summary.scalar(
                    f"train/contrastive_acc_top{k}/bottleneck_concat",
                    bottle_topk_acc[k],
                    step=global_step)

            if loss_name == "ntxent":
              for i_partial in range(len(partial_embeddings)):
                for i_head in range(n_projection_heads):
                  tf2.summary.scalar(
                      f"train/overview/ntxent_loss_{i_partial}_{i_head}",
                      contrast_losses[i_partial * n_projection_heads + i_head],
                      step=global_step)

              tf2.summary.scalar(
                  "train/overview/ntxent_loss",
                  tf.reduce_sum(contrast_losses),
                  step=global_step)

            if loss_name == "simsiam":
              for i_head, contrast_loss in enumerate(contrast_losses):
                tf2.summary.scalar(
                    f"train/overview/simsiam_loss_{i_head}",
                    contrast_loss,
                    step=global_step)

            tf2.summary.scalar(
                "train/overview/corr_loss", corr_loss, step=global_step)

            if distance_head:
              tf2.summary.scalar(
                  "train/overview/distance_loss",
                  tf.reduce_sum(distance_losses),
                  step=global_step)

            tf2.summary.scalar(
                "train/overview/total_loss", loss, step=global_step)

          summary_writer.flush()

      control_deps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      control_deps.extend(tf.summary.all_v2_summary_ops())
      optimizer = lars_optimizer.LARSOptimizer(
          learning_rate_step,
          momentum=momentum,
          weight_decay=weight_decay,
          exclude_from_weight_decay=["bn", "bias"])

      if use_tpu:
        optimizer = tf.tpu.CrossShardOptimizer(optimizer)

      with tf.control_dependencies(control_deps):
        train_op = optimizer.minimize(
            loss, global_step=global_step, var_list=tf.trainable_variables())

      return tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, train_op=train_op, loss=loss)

  if distance_bin_bounds is None:
    distance_bin_bounds = [0, 1e12]

  return _model_fn

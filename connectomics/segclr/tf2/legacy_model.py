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
"""Legacy (tf1) model support for loading and refinement training."""

from typing import Any
from connectomics.segclr import model_util
from connectomics.segclr import resnet
import tensorflow as tf


class LegacySegClrModel(tf.keras.layers.Layer):
  """Model builder for SegCLR models based on TensorFlow 1.

  The SegCLR models in the current paper were trained via TensorFlow 1.  New
  TensorFlow 2 versions are under development, so for compatibility we provide
  a shim to load the legacy TF1 models into TF2.
  """

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, features: Any, training: bool = True) -> tuple[Any, None]:
    with tf.compat.v1.variable_scope('base_model_0'):
      rs = resnet.resnet_stack(
          'resnet18', features, batch_norm=True, is_training=training
      )

    with tf.compat.v1.variable_scope('bottleneck_head_0'):
      rs = model_util.projection_head(
          rs, out_dim=64, num_layers=3, is_training=training, use_bn=True
      )

    if not training:
      return rs, None

    with tf.compat.v1.variable_scope('projection_head_0_0'):
      rs = model_util.projection_head(
          rs, out_dim=16, num_layers=3, is_training=training, use_bn=True
      )

    # Match return to tf2 model
    return rs, None


def restore_legacy_model(chkpt_path: str) -> LegacySegClrModel:
  """Loads SegCLR legacy model."""
  legacy_model = LegacySegClrModel()

  # Build model
  mock_input = tf.keras.Input(
      shape=(129, 129, 129, 1), batch_size=32, dtype=tf.float32
  )
  _ = legacy_model(mock_input)

  # Load checkpoint
  ckpt = tf.train.Checkpoint(legacy_model)
  _ = ckpt.restore(chkpt_path).assert_existing_objects_matched()

  return legacy_model

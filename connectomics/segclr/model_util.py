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
"""Supports model.py."""
from simclr.resnet import BatchNormalization
import tensorflow.compat.v1 as tf

DEFAULT_EPSILON = 1e-5


def linear_layer(x: tf.Tensor, is_training: bool, num_outputs: int,
                 use_bias: bool = True, use_bn: bool = False,
                 batch_norm_decay: float = 0.9, use_relu: bool = False,
                 name: str = "linear_layer") -> tf.Tensor:
  """Linear layer.

  Args:
    x: hidden state tensor of shape (batch size, dim).
    is_training: boolean indicator for training or test.
    num_outputs: number of classes.
    use_bias: whether or not to use bias.
    use_bn: whether or not to use BN for output units.
    batch_norm_decay: batch norm decay
    use_relu: whether to apply relu
    name: the name for variable scope.

  Returns:
    logits of shape (batch size, num_outputs)
  """
  assert x.shape.ndims == 2, x.shape
  with tf.variable_scope(name):
    x = tf.layers.dense(
        inputs=x, units=num_outputs, use_bias=use_bias and not use_bn,
        kernel_initializer=tf.random_normal_initializer(stddev=.01))

    if use_bn:
      bn_f = BatchNormalization(
          momentum=batch_norm_decay, epsilon=DEFAULT_EPSILON, renorm=False,
          fused=False, axis=-1, name=f"{name}_bn")
      x = bn_f(x, training=is_training)
    x = tf.identity(x, "%s_out" % name)

    if use_relu:
      x = tf.nn.relu(x)
  return x


def projection_head(x: tf.Tensor, out_dim: int = 128, num_layers: int = 2,
                    is_training: bool = True, use_bn: bool = False,
                    batch_norm_decay: float = 0.9) -> tf.Tensor:
  """Projection head.

  Args:
    x: input, batch size x N features
    out_dim: output dimensions
    num_layers: number of layers
    is_training: are we training?
    use_bn: whether to use batch norm
    batch_norm_decay: batch norm decay

  Returns:
    x: head output
  """
  x_dim = x.shape[-1]
  x_list = [x]

  for i_layer in range(num_layers):
    if i_layer != num_layers - 1:
      # for the middle layers, use bias and relu for the output.
      dim, bias_relu = x_dim, True
    else:
      # for the final layer, neither bias nor relu is used.
      dim, bias_relu = out_dim, False

    x = linear_layer(
        x, is_training, dim, use_bias=bias_relu, use_relu=bias_relu,
        use_bn=use_bn, batch_norm_decay=batch_norm_decay, name=f"nl_{i_layer}")

    x_list.append(x)

  return x


def simsiam_projection_head(x: tf.Tensor, is_training: bool = True,
                            use_bn: bool = False, batch_norm_decay: float = 0.9,
                            simsiam_df: int = 4) -> tf.Tensor:
  """Projection head.

  Simsiam specific.

  Args:
    x: input, batch size x N features
    is_training: are we training?
    use_bn: whether to use batch norm
    batch_norm_decay: batch norm decay
    simsiam_df: reduction factor in simsiam head

  Returns:
    x: output, batch size x M features

  """

  in_dim = x.shape[-1]
  bottle_dim = in_dim // simsiam_df

  x = linear_layer(
      x, is_training, bottle_dim, use_bias=False, use_relu=True, use_bn=use_bn,
      batch_norm_decay=batch_norm_decay, name="nl_1")

  x = linear_layer(
      x, is_training, in_dim, use_bias=False, use_relu=False, use_bn=False,
      batch_norm_decay=batch_norm_decay, name="nl_2")

  return x


def learning_rate_step(base_learning_rate: float, warmup_steps: int = 0,
                       total_steps: int = 500000)-> tf.Tensor:
  """Build learning rate schedule."""

  global_step = tf.train.get_or_create_global_step()

  # Cosine decay learning rate schedule
  learning_rate = tf.where(
      global_step < warmup_steps,
      base_learning_rate,
      tf.train.cosine_decay(
          learning_rate=base_learning_rate,
          global_step=global_step - warmup_steps,
          decay_steps=total_steps - warmup_steps,
          alpha=0.0))

  return learning_rate

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
"""Contains definitions for the post-activation form of Residual Networks.

Adapted for 3D from simclr/tf2/resnet.py

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from simclr.tf2.resnet import BatchNormRelu
from simclr.tf2.resnet import DropBlock
import tensorflow.compat.v2 as tf


class FixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self, kernel_size, data_format='channels_last', **kwargs):
    super().__init__(**kwargs)
    self.kernel_size = kernel_size
    self.data_format = data_format

  def call(self, inputs, training):
    kernel_size = self.kernel_size
    data_format = self.data_format
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
      padded_inputs = tf.pad(
          inputs,
          [
              [0, 0],
              [0, 0],
              [pad_beg, pad_end],
              [pad_beg, pad_end],
              [pad_beg, pad_end],
          ],
      )
    else:
      padded_inputs = tf.pad(
          inputs,
          [
              [0, 0],
              [pad_beg, pad_end],
              [pad_beg, pad_end],
              [pad_beg, pad_end],
              [0, 0],
          ],
      )

    return padded_inputs


class Conv3dFixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring
  """Convolution with padding fixes for strides > 1.

  Adds additional padding before applying the convolution to enable same shapes
  even with strides > 1. This behavior is copied from SimCLR.
  """

  def __init__(
      self, filters, kernel_size, strides, data_format='channels_last', **kwargs
  ):
    super().__init__(**kwargs)
    if strides > 1:
      self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
    else:
      self.fixed_padding = None
    self.conv3d = tf.keras.layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        data_format=data_format,
    )

  def call(self, inputs, training):
    if self.fixed_padding:
      inputs = self.fixed_padding(inputs, training=training)
    return self.conv3d(inputs, training=training)


class IdentityLayer(tf.keras.layers.Layer):

  def call(self, inputs, training):
    return tf.identity(inputs)


class ResidualBlock(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(
      self,
      filters,
      strides,
      use_projection=False,
      data_format='channels_last',
      dropblock_keep_prob=None,
      dropblock_size=None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    del dropblock_keep_prob
    del dropblock_size
    self.conv3d_bn_layers = []
    self.shortcut_layers = []
    if use_projection:
      self.shortcut_layers.append(
          Conv3dFixedPadding(
              filters=filters,
              kernel_size=1,
              strides=strides,
              data_format=data_format,
          )
      )
      self.shortcut_layers.append(
          BatchNormRelu(relu=False, data_format=data_format)
      )

    self.conv3d_bn_layers.append(
        Conv3dFixedPadding(
            filters=filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format,
        )
    )
    self.conv3d_bn_layers.append(BatchNormRelu(data_format=data_format))
    self.conv3d_bn_layers.append(
        Conv3dFixedPadding(
            filters=filters, kernel_size=3, strides=1, data_format=data_format
        )
    )
    self.conv3d_bn_layers.append(
        BatchNormRelu(
            relu=False,
            init_zero=True,
            data_format=data_format,
        )
    )

  def call(self, inputs, training):
    shortcut = inputs
    for layer in self.shortcut_layers:
      # Projection shortcut in first layer to match filters and strides
      shortcut = layer(shortcut, training=training)

    for layer in self.conv3d_bn_layers:
      inputs = layer(inputs, training=training)

    return tf.nn.relu(inputs + shortcut)


class BottleneckBlock(tf.keras.layers.Layer):
  """BottleneckBlock."""

  def __init__(
      self,
      filters,
      strides,
      use_projection=False,
      data_format='channels_last',
      dropblock_keep_prob=None,
      dropblock_size=None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.projection_layers = []
    if use_projection:
      filters_out = 4 * filters
      self.projection_layers.append(
          Conv3dFixedPadding(
              filters=filters_out,
              kernel_size=1,
              strides=strides,
              data_format=data_format,
          )
      )
      self.projection_layers.append(
          BatchNormRelu(relu=False, data_format=data_format)
      )
    self.shortcut_dropblock = DropBlock(
        data_format=data_format,
        keep_prob=dropblock_keep_prob,
        dropblock_size=dropblock_size,
    )

    self.conv_relu_dropblock_layers = []

    self.conv_relu_dropblock_layers.append(
        Conv3dFixedPadding(
            filters=filters, kernel_size=1, strides=1, data_format=data_format
        )
    )
    self.conv_relu_dropblock_layers.append(
        BatchNormRelu(data_format=data_format)
    )
    self.conv_relu_dropblock_layers.append(
        DropBlock(
            data_format=data_format,
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
        )
    )

    self.conv_relu_dropblock_layers.append(
        Conv3dFixedPadding(
            filters=filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format,
        )
    )
    self.conv_relu_dropblock_layers.append(
        BatchNormRelu(data_format=data_format)
    )
    self.conv_relu_dropblock_layers.append(
        DropBlock(
            data_format=data_format,
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
        )
    )

    self.conv_relu_dropblock_layers.append(
        Conv3dFixedPadding(
            filters=4 * filters,
            kernel_size=1,
            strides=1,
            data_format=data_format,
        )
    )
    self.conv_relu_dropblock_layers.append(
        BatchNormRelu(
            relu=False,
            init_zero=True,
            data_format=data_format,
        )
    )
    self.conv_relu_dropblock_layers.append(
        DropBlock(
            data_format=data_format,
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
        )
    )

  def call(self, inputs, training):
    shortcut = inputs
    for layer in self.projection_layers:
      shortcut = layer(shortcut, training=training)
    shortcut = self.shortcut_dropblock(shortcut, training=training)

    for layer in self.conv_relu_dropblock_layers:
      inputs = layer(inputs, training=training)

    return tf.nn.relu(inputs + shortcut)


class BlockGroup(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(
      self,
      filters,
      block_fn,
      blocks,
      strides,
      data_format='channels_last',
      dropblock_keep_prob=None,
      dropblock_size=None,
      **kwargs,
  ):
    self._name = kwargs.get('name')
    super().__init__(**kwargs)

    self.layers = []
    self.layers.append(
        block_fn(
            filters,
            strides,
            use_projection=True,
            data_format=data_format,
            dropblock_keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
        )
    )

    for _ in range(1, blocks):
      self.layers.append(
          block_fn(
              filters,
              1,
              data_format=data_format,
              dropblock_keep_prob=dropblock_keep_prob,
              dropblock_size=dropblock_size,
          )
      )

  def call(self, inputs, training):
    for layer in self.layers:
      inputs = layer(inputs, training=training)
    return tf.identity(inputs, self._name)


class Resnet(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(
      self,
      block_fn,
      layers,
      width_multiplier,
      cifar_stem=False,
      data_format='channels_last',
      dropblock_keep_probs=None,
      dropblock_size=None,
      train_mode='pretrain',
      fine_tune_after_block=-1,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.data_format = data_format
    self.train_mode = train_mode
    self.fine_tune_after_block = fine_tune_after_block
    if dropblock_keep_probs is None:
      dropblock_keep_probs = [None] * 4
    if (
        not isinstance(dropblock_keep_probs, list)
        or len(dropblock_keep_probs) != 4
    ):
      raise ValueError(
          'dropblock_keep_probs is not valid:', dropblock_keep_probs
      )
    trainable = (
        self.train_mode != 'finetune' or self.fine_tune_after_block == -1
    )
    self.initial_conv_relu_max_pool = []
    if cifar_stem:
      self.initial_conv_relu_max_pool.append(
          Conv3dFixedPadding(
              filters=64 * width_multiplier,
              kernel_size=3,
              strides=1,
              data_format=data_format,
              trainable=trainable,
          )
      )
      self.initial_conv_relu_max_pool.append(
          IdentityLayer(name='initial_conv', trainable=trainable)
      )
      self.initial_conv_relu_max_pool.append(
          BatchNormRelu(data_format=data_format, trainable=trainable)
      )
      self.initial_conv_relu_max_pool.append(
          IdentityLayer(name='initial_max_pool', trainable=trainable)
      )
    else:
      self.initial_conv_relu_max_pool.append(
          Conv3dFixedPadding(
              filters=64 * width_multiplier,
              kernel_size=7,
              strides=2,
              data_format=data_format,
              trainable=trainable,
          )
      )
      self.initial_conv_relu_max_pool.append(
          IdentityLayer(name='initial_conv', trainable=trainable)
      )
      self.initial_conv_relu_max_pool.append(
          BatchNormRelu(data_format=data_format, trainable=trainable)
      )

      self.initial_conv_relu_max_pool.append(
          tf.keras.layers.MaxPooling3D(
              pool_size=3,
              strides=2,
              padding='SAME',
              data_format=data_format,
              trainable=trainable,
          )
      )
      self.initial_conv_relu_max_pool.append(
          IdentityLayer(name='initial_max_pool', trainable=trainable)
      )

    self.block_groups = []
    if self.train_mode == 'finetune' and self.fine_tune_after_block == 0:
      trainable = True

    self.block_groups.append(
        BlockGroup(
            filters=64 * width_multiplier,
            block_fn=block_fn,
            blocks=layers[0],
            strides=1,
            name='block_group1',
            data_format=data_format,
            dropblock_keep_prob=dropblock_keep_probs[0],
            dropblock_size=dropblock_size,
            trainable=trainable,
        )
    )

    if self.train_mode == 'finetune' and self.fine_tune_after_block == 1:
      trainable = True

    self.block_groups.append(
        BlockGroup(
            filters=128 * width_multiplier,
            block_fn=block_fn,
            blocks=layers[1],
            strides=2,
            name='block_group2',
            data_format=data_format,
            dropblock_keep_prob=dropblock_keep_probs[1],
            dropblock_size=dropblock_size,
            trainable=trainable,
        )
    )

    if self.train_mode == 'finetune' and self.fine_tune_after_block == 2:
      trainable = True

    self.block_groups.append(
        BlockGroup(
            filters=256 * width_multiplier,
            block_fn=block_fn,
            blocks=layers[2],
            strides=2,
            name='block_group3',
            data_format=data_format,
            dropblock_keep_prob=dropblock_keep_probs[2],
            dropblock_size=dropblock_size,
            trainable=trainable,
        )
    )

    if self.train_mode == 'finetune' and self.fine_tune_after_block == 3:
      trainable = True

    self.block_groups.append(
        BlockGroup(
            filters=512 * width_multiplier,
            block_fn=block_fn,
            blocks=layers[3],
            strides=2,
            name='block_group4',
            data_format=data_format,
            dropblock_keep_prob=dropblock_keep_probs[3],
            dropblock_size=dropblock_size,
            trainable=trainable,
        )
    )

    if self.train_mode == 'finetune' and self.fine_tune_after_block == 4:
      # This case doesn't really matter.
      trainable = True

  def call(self, inputs, training):
    for layer in self.initial_conv_relu_max_pool:
      inputs = layer(inputs, training=training)

    for i, layer in enumerate(self.block_groups):
      if self.train_mode == 'finetune' and self.fine_tune_after_block == i:
        inputs = tf.stop_gradient(inputs)
      inputs = layer(inputs, training=training)
    if self.train_mode == 'finetune' and self.fine_tune_after_block == 4:
      inputs = tf.stop_gradient(inputs)
    if self.data_format == 'channels_last':
      inputs = tf.reduce_mean(inputs, [1, 2, 3])
    else:
      inputs = tf.reduce_mean(inputs, [2, 3, 4])

    inputs = tf.identity(inputs, 'final_avg_pool')
    return inputs


def resnet(
    resnet_depth,
    width_multiplier,
    fine_tune_after_block=-1,
    train_mode='pretrain',
    cifar_stem=False,
    data_format='channels_last',
    dropblock_keep_probs=None,
    dropblock_size=None,
):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
      34: {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
      50: {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},
      101: {'block': BottleneckBlock, 'layers': [3, 4, 23, 3]},
      152: {'block': BottleneckBlock, 'layers': [3, 8, 36, 3]},
      200: {'block': BottleneckBlock, 'layers': [3, 24, 36, 3]},
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return Resnet(
      params['block'],
      params['layers'],
      width_multiplier,
      train_mode=train_mode,
      cifar_stem=cifar_stem,
      dropblock_keep_probs=dropblock_keep_probs,
      dropblock_size=dropblock_size,
      fine_tune_after_block=fine_tune_after_block,
      data_format=data_format,
  )

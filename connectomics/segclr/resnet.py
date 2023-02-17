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
"""Resnet compiled from code in tensorflow/fragments.

The main difference is the addition of TPU pod compatible batchnorm.
"""
import functools
from typing import List
from connectomics.segclr import model_util
import tensorflow.compat.v1 as tf

DEFAULT_MOMENTUM = 0.99
DEFAULT_EPSILON = 1e-5


def _resnet_block(inputs: tf.Tensor,
                  filters: List[int],
                  kernel_sizes: List[int],
                  strides: List[int],
                  conv_shortcut: bool = False,
                  batch_norm: bool = True,
                  batch_norm_decay: float = 0.9,
                  is_training: bool = True,
                  block: str = '') -> tf.Tensor:
  """A 3D residual network building block.

  Reference: - [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385)
  Adapted from Keras Applications ResNet50.

  A resnet building block primarily varies in whether it's the first layer of a
  stage or not. A stage consists of blocks with same feature depth. If it's the
  first layer in a stage, it needs a convolution op in skip connection to adapt
  to the feature depth change, otherwise no op is needed.

  Arguments:
    inputs: A 5D tensor with shape=(batch_size, z, y, x, channels).
    filters: A list of integers, specifying the filter number of each conv layer
      at main path. e.g. (64, 64) for resnet 18 and (64, 64, 256) for resnet 50.
    kernel_sizes: The kernel sizes of each conv layers at main path, e.g. (3, 3)
      for resnet 18 and (1, 3, 3) for resnet 50.
    strides: Strides for each conv layer at the main path. Typically the first
      layer in 'conv' needs to have stride=2, thus (2, 1) for 'conv', (1, 1) for
      'identity' in resnet18, (2, 1, 1) for 'conv', (1, 1, 1) for 'identity' in
      resnet50.
    conv_shortcut: Controls whether the shortcut of this block uses 1x1x1
      convolution, typically the first block of a new feature depth stage in
      resnet needs to have the shortcut path convoluted to adapt to the change
      of feature depth and striding in main path.
    batch_norm: Whether to use batch normalization.
    batch_norm_decay: batch norm decay
    is_training: Whether in training mode.
    block: 'a','b'..., current block label, used for generating layer names.

  Returns:
    Output tensor for the block.
  """
  assert len(filters) == len(kernel_sizes) == len(strides)
  with tf.variable_scope('block_%s' % block, reuse=tf.AUTO_REUSE):
    x = inputs
    # main branch
    for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
      x = tf.layers.conv3d(
          x,
          filters=f,
          kernel_size=k,
          strides=s,
          padding='SAME',
          name='conv_%d' % i)
      x = tf.nn.relu(x, name='relu_%d' % i)
      if batch_norm:
        bn_f = model_util.BatchNormalization(
            momentum=batch_norm_decay,
            epsilon=DEFAULT_EPSILON,
            renorm=True,
            fused=False,
            axis=-1,
            name='bn_%d' % i)
        x = bn_f(x, training=is_training)
    # shortcut branch
    if conv_shortcut:
      shortcut = tf.layers.conv3d(
          inputs,
          filters=filters[-1],
          kernel_size=(1, 1, 1),
          strides=strides[0],
          padding='SAME',
          name='shortcut_conv')
      if batch_norm:
        bn_f = model_util.BatchNormalization(
            momentum=batch_norm_decay,
            epsilon=DEFAULT_EPSILON,
            renorm=True,
            fused=False,
            axis=-1,
            name='shortcut_bn')
        shortcut = bn_f(shortcut, training=is_training)
    else:
      shortcut = inputs

    x = tf.add(x, shortcut, name='add')
    x = tf.nn.relu(x, name='relu')
    return x


def resnet_stack(resnet_kind: str,
                 patches: tf.Tensor,
                 batch_norm: bool = False,
                 batch_norm_decay: float = 0.9,
                 is_training: bool = True) -> tf.Tensor:
  """ResNet stack selector.

  Args:
    resnet_kind: ResNet definition; has to be one of resnet18,50,101
    patches: A 5D tensor in NDHWC format, representing input patches
    batch_norm: Whether to use batch norm layers
    batch_norm_decay: batch norm decay
    is_training: are we training

  Returns:

  """
  if resnet_kind == 'resnet18':
    return resnet18_stack(patches, batch_norm, batch_norm_decay, is_training)
  elif resnet_kind == 'resnet50':
    return resnet50_stack(patches, batch_norm, batch_norm_decay, is_training)
  elif resnet_kind == 'resnet101':
    return resnet101_stack(patches, batch_norm, batch_norm_decay, is_training)
  else:
    raise NotImplementedError('Unknown resnet kind')


def resnet18_stack(patches: tf.Tensor,
                   batch_norm: bool = False,
                   batch_norm_decay: float = 0.9,
                   is_training: bool = False) -> tf.Tensor:
  """A 3D adaptation of ResNet18.

  Args:
    patches: A 5D tensor in NDHWC format, representing input patches
    batch_norm: Whether to use batch norm layers
    batch_norm_decay: batch norm decay
    is_training: Whether in inference or training mode

  Returns:
    Output logits

  """

  conv = functools.partial(
      _resnet_block,
      kernel_sizes=[3, 3],
      strides=[2, 1],
      conv_shortcut=True,
      batch_norm=batch_norm,
      batch_norm_decay=batch_norm_decay,
      is_training=is_training)
  identity = functools.partial(
      _resnet_block,
      kernel_sizes=[3, 3],
      strides=[1, 1],
      conv_shortcut=False,
      batch_norm=batch_norm,
      batch_norm_decay=batch_norm_decay,
      is_training=is_training)

  with tf.variable_scope('resnet18', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('stage_1', reuse=tf.AUTO_REUSE):
      x = tf.layers.conv3d(
          patches,
          filters=64,
          kernel_size=7,
          strides=2,
          padding='SAME',
          name='conv1')
      if batch_norm:
        bn_f = model_util.BatchNormalization(
            momentum=batch_norm_decay,
            epsilon=DEFAULT_EPSILON,
            renorm=False,
            fused=False,
            axis=-1,
            name='bn1')
        x = bn_f(x, training=is_training,)
      x = tf.layers.max_pooling3d(x, 3, strides=2, padding='SAME', name='pool1')
    with tf.variable_scope('stage_2', reuse=tf.AUTO_REUSE):
      x = _resnet_block(
          x,
          filters=[64, 64],
          kernel_sizes=[3, 3],
          strides=[1, 1],
          conv_shortcut=True,
          batch_norm=batch_norm,
          batch_norm_decay=batch_norm_decay,
          is_training=is_training,
          block='a')
      x = identity(x, filters=[64, 64], block='b')

    with tf.variable_scope('stage_3', reuse=tf.AUTO_REUSE):
      x = conv(x, filters=[128, 128], block='a')
      x = identity(x, filters=[128, 128], block='b')

    with tf.variable_scope('stage_4', reuse=tf.AUTO_REUSE):
      x = conv(x, filters=[256, 256], block='a')
      x = identity(x, filters=[256, 256], block='b')

    with tf.variable_scope('stage_5', reuse=tf.AUTO_REUSE):
      x = conv(x, filters=[512, 512], block='a')
      x = identity(x, filters=[512, 512], block='b')

      x = tf.keras.layers.GlobalAveragePooling3D(name='avg_pool')(x)

    return x


def resnet50_stack(patches: tf.Tensor,
                   batch_norm: bool = False,
                   batch_norm_decay: float = 0.9,
                   is_training: bool = True) -> tf.Tensor:
  """A 3D adaptation of ResNet50.

  Args:
    patches: A 5D tensor in NDHWC format, representing input patches
    batch_norm: Whether to use batch norm layers
    batch_norm_decay: batch norm decay
    is_training: Whether in inference or training mode

  Returns:
    Output logits

  """
  conv = functools.partial(
      _resnet_block,
      kernel_sizes=[1, 3, 1],
      strides=[2, 1, 1],
      conv_shortcut=True,
      batch_norm=batch_norm,
      batch_norm_decay=batch_norm_decay,
      is_training=is_training)
  identity = functools.partial(
      _resnet_block,
      kernel_sizes=[1, 3, 1],
      strides=[1, 1, 1],
      conv_shortcut=False,
      batch_norm=batch_norm,
      batch_norm_decay=batch_norm_decay,
      is_training=is_training)
  with tf.variable_scope('resnet50'):
    with tf.variable_scope('stage_1'):
      x = tf.layers.conv3d(
          patches,
          filters=64,
          kernel_size=7,
          strides=2,
          padding='SAME',
          name='conv1')
      if batch_norm:
        bn_f = model_util.BatchNormalization(
            momentum=batch_norm_decay,
            epsilon=DEFAULT_EPSILON,
            renorm=False,
            fused=False,
            axis=-1,
            name='bn1')
        x = bn_f(x, training=is_training)
      x = tf.layers.max_pooling3d(x, 3, strides=2, padding='SAME', name='pool1')
    with tf.variable_scope('stage_2'):
      # strides is set to 1 for the first block after maxpooling
      x = _resnet_block(
          x,
          filters=[64, 64, 256],
          kernel_sizes=[1, 3, 1],
          strides=[1, 1, 1],
          conv_shortcut=True,
          batch_norm=batch_norm,
          batch_norm_decay=batch_norm_decay,
          is_training=is_training,
          block='a')
      for block in ['b', 'c']:
        x = identity(x, filters=[64, 64, 256], block=block)

    with tf.variable_scope('stage_3'):
      x = conv(x, filters=[128, 128, 512], block='a')
      for block in ['b', 'c', 'd']:
        x = identity(x, filters=[128, 128, 512], block=block)

    with tf.variable_scope('stage_4'):
      x = conv(x, filters=[256, 256, 1024], block='a')
      for block in ['b', 'c', 'd', 'e', 'f']:
        x = identity(x, filters=[256, 256, 1024], block=block)

    with tf.variable_scope('stage_5'):
      x = conv(x, filters=[512, 512, 2048], block='a')
      for block in ['b', 'c']:
        x = identity(x, filters=[512, 512, 2048], block=block)

      x = tf.keras.layers.GlobalAveragePooling3D(name='avg_pool')(x)

    return x


def resnet101_stack(patches: tf.Tensor,
                    batch_norm: bool = True,
                    batch_norm_decay: float = 0.9,
                    is_training: bool = True) -> tf.Tensor:
  """A 3D adaptation of ResNet101.

  Args:
    patches: A 5D tensor in NDHWC format, representing input patches
    batch_norm: Whether to use batch norm layers
    batch_norm_decay: batch norm decay
    is_training: Whether in inference or training mode

  Returns:
    Final AvgPool layer, before logits or projection heads.
  """
  conv = functools.partial(
      _resnet_block,
      kernel_sizes=[1, 3, 1],
      strides=[2, 1, 1],
      conv_shortcut=True,
      batch_norm=batch_norm,
      batch_norm_decay=batch_norm_decay,
      is_training=is_training)
  identity = functools.partial(
      _resnet_block,
      kernel_sizes=[1, 3, 1],
      strides=[1, 1, 1],
      conv_shortcut=False,
      batch_norm=batch_norm,
      batch_norm_decay=batch_norm_decay,
      is_training=is_training)
  with tf.variable_scope('resnet101'):
    with tf.variable_scope('stage_1'):
      x = tf.layers.conv3d(
          patches,
          filters=64,
          kernel_size=7,
          strides=2,
          padding='SAME',
          name='conv1')
      if batch_norm:
        bn_f = model_util.BatchNormalization(
            momentum=batch_norm_decay,
            epsilon=DEFAULT_EPSILON,
            renorm=False,
            fused=False,
            axis=-1,
            name='bn1')
        x = bn_f(x, training=is_training)
      x = tf.layers.max_pooling3d(x, 3, strides=2, padding='SAME', name='pool1')
    with tf.variable_scope('stage_2'):
      # strides is set to 1 for the first block after maxpooling
      x = _resnet_block(
          x,
          filters=[64, 64, 256],
          kernel_sizes=[1, 3, 1],
          strides=[1, 1, 1],
          conv_shortcut=True,
          batch_norm=batch_norm,
          batch_norm_decay=batch_norm_decay,
          is_training=is_training,
          block='a')
      for block in ['b', 'c']:
        x = identity(x, filters=[64, 64, 256], block=block)

    with tf.variable_scope('stage_3'):
      x = conv(x, filters=[128, 128, 512], block='a')
      for block in ['b', 'c', 'd']:
        x = identity(x, filters=[128, 128, 512], block=block)

    with tf.variable_scope('stage_4'):
      x = conv(x, filters=[256, 256, 1024], block='a')
      for block in [str(i) for i in range(23)]:
        x = identity(x, filters=[256, 256, 1024], block=block)

    with tf.variable_scope('stage_5'):
      x = conv(x, filters=[512, 512, 2048], block='a')
      for block in ['b', 'c']:
        x = identity(x, filters=[512, 512, 2048], block=block)

      x = tf.keras.layers.GlobalAveragePooling3D(name='avg_pool')(x)

    return x

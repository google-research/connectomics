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
"""Model specification for SimCLR.

Adapted for SegCLR from simclr/tf2/model.py

Required FLAGS for lower level simclr implementation:
global_bn: bool, default: True
BatchNorm is used either way for model training. global_bn synchronizes batch
statistics across distributed devices. In SegCLR's tf1 implementation this was
also used when batch norm was activated. Recommended to set to True.

batch_norm_decay: float, default: 0.9
This is ultimately passed as the momentum argument to the batch norm layer.
See here for more info:
https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/SyncBatchNormalization
"""

from absl import flags
from connectomics.segclr.tf2 import resnet
from simclr.tf2 import lars_optimizer
from simclr.tf2.model import LinearLayer
from simclr.tf2.model import SupervisedHead
import tensorflow.compat.v2 as tf

# global_bn and batch_norm_decay are defined here as flags for compatibility
# with upstream modules in SimCLR.
flags.DEFINE_boolean(
    'global_bn',
    True,
    'Whether to aggregate BN statistics across distributed cores.',
)
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')


def build_optimizer(learning_rate, momentum, weight_decay):
  return lars_optimizer.LARSOptimizer(
      learning_rate,
      momentum=momentum,
      weight_decay=weight_decay,
      exclude_from_weight_decay=[
          'batch_normalization',
          'bias',
          'head_supervised',
      ],
  )


class ProjectionHead(tf.keras.layers.Layer):
  """Multi-layered head for SegCLR embedding model."""

  def __init__(
      self,
      proj_out_dim: int,
      num_proj_layers: int,
      proj_head_mode: str = 'nonlinear',
      layer_name_prefix: str = 'proj',
      ft_proj_selector: int = 0,
      **kwargs,
  ):
    """Initializer.

    Args:
      proj_out_dim: number of projection dimensions
      num_proj_layers: number of projection layers
      proj_head_mode: projection head type: ['none', 'linear', 'nonlinear']
      layer_name_prefix: prefix for layer names
      ft_proj_selector: feature selector for second output; matches projection
        head layer.
      **kwargs: additional keyword arguments

    Returns:
      projection head output
      selected projection layer features
    """
    self.proj_head_mode = proj_head_mode
    self.num_proj_layers = num_proj_layers
    self.ft_proj_selector = ft_proj_selector

    self.linear_layers = []
    if proj_head_mode == 'none':
      pass  # directly use the output hiddens as hiddens
    elif proj_head_mode == 'linear':
      self.linear_layers = [
          LinearLayer(
              num_classes=proj_out_dim,
              use_bias=False,
              use_bn=True,
              name=f'{layer_name_prefix}_l_0',
          )
      ]
    elif proj_head_mode == 'nonlinear':
      for j in range(self.num_proj_layers):
        if j != num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          self.linear_layers.append(
              LinearLayer(
                  num_classes=lambda input_shape: int(input_shape[-1]),
                  use_bias=True,
                  use_bn=True,
                  name=f'{layer_name_prefix}_nl_{j}',
              )
          )
        else:
          # for the final layer, neither bias nor relu is used.
          self.linear_layers.append(
              LinearLayer(
                  num_classes=proj_out_dim,
                  use_bias=False,
                  use_bn=True,
                  name=f'{layer_name_prefix}_nl_{j}',
              )
          )
    else:
      raise ValueError('Unknown head projection mode {}'.format(proj_head_mode))
    super().__init__(**kwargs)

  def call(self, inputs, training):
    if self.proj_head_mode == 'none':
      return inputs, None  # directly use the output hiddens as hiddens
    hiddens_list = [tf.identity(inputs, 'proj_head_input')]
    if self.proj_head_mode == 'linear':
      assert len(self.linear_layers) == 1, len(self.linear_layers)
      hiddens_list.append(
          self.linear_layers[0](hiddens_list[-1], training)
      )
    elif self.proj_head_mode == 'nonlinear':
      for j in range(self.num_proj_layers):
        hiddens = self.linear_layers[j](hiddens_list[-1], training)
        if j != self.num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          hiddens = tf.nn.relu(hiddens)
        hiddens_list.append(hiddens)
    else:
      raise ValueError(
          'Unknown head projection mode {}'.format(self.proj_head_mode)
      )
    # The first element is the output of the projection head.
    # The second element is the input of the finetune head.
    proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
    return proj_head_output, hiddens_list[self.ft_proj_selector]


class Model(tf.keras.models.Model):
  """Resnet model with bottleneck layers and projection or supervised layer."""

  def __init__(
      self,
      resnet_depth,
      proj_out_dim,
      num_proj_layers,
      bottleneck_dim,
      num_bottleneck_layers,
      proj_head_mode='nonlinear',
      train_mode='pretrain',
      width_multiplier=1,
      num_transforms=2,
      lineareval_while_pretraining=False,
      fine_tune_after_block=-1,
      use_blur=False,
      num_classes=1,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.num_transforms = num_transforms
    self.fine_tune_after_block = fine_tune_after_block
    self.use_blur = use_blur
    self.train_mode = train_mode
    self.lineareval_while_pretraining = lineareval_while_pretraining
    self.resnet_model = resnet.resnet(
        resnet_depth=resnet_depth,
        width_multiplier=width_multiplier,
        train_mode=self.train_mode,
        fine_tune_after_block=self.fine_tune_after_block,
        cifar_stem=False,
    )
    self._bottleneck_head = ProjectionHead(
        bottleneck_dim, num_bottleneck_layers, 'nonlinear', name='bottleneck'
    )
    self._projection_head = ProjectionHead(
        proj_out_dim, num_proj_layers, proj_head_mode
    )
    if self.train_mode == 'finetune' or self.lineareval_while_pretraining:
      self.supervised_head = SupervisedHead(num_classes)

  def __call__(self, features, training):
    if training and self.train_mode == 'pretrain':
      if self.fine_tune_after_block > -1:
        raise ValueError(
            'Does not support layer freezing during pretraining,'
            'should set fine_tune_after_block<=-1 for safety.'
        )

    # Base network forward pass.
    hiddens = self._bottleneck_head(
        self.resnet_model(features, training=training)
    )[0]

    # Add heads.
    projection_head_outputs, supervised_head_inputs = self._projection_head(
        hiddens, training
    )

    if self.train_mode == 'finetune':
      supervised_head_outputs = self.supervised_head(
          supervised_head_inputs, training
      )
      return None, supervised_head_outputs
    elif self.train_mode == 'pretrain' and self.lineareval_while_pretraining:
      # When performing pretraining and linear evaluation together we do not
      # want information from linear eval flowing back into pretraining network
      # so we put a stop_gradient.
      supervised_head_outputs = self.supervised_head(
          tf.stop_gradient(supervised_head_inputs), training
      )
      return projection_head_outputs, supervised_head_outputs
    else:
      return projection_head_outputs, None

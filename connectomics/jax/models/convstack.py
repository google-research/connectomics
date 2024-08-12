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
"""2d/3d residual convstack."""

import itertools
from typing import Iterable

from connectomics.common.bounding_box import BoundingBox
from connectomics.jax import parameter_replacement_util as param_util
from connectomics.jax import util
from flax import struct
import flax.linen as nn
import jax.numpy as jnp


@struct.dataclass
class ConvstackConfig:
  """Config settings for residual convstacks.

  Attributes:
    features: number of feature maps
    depth: number of residual modules
    padding: padding mode to use for convolutions ('same', 'valid')
    dim: number of spatial dimensions
    num_convs: number of convolutions in the residual module
    use_layernorm: whether to use layer normalization; this has been observed to
      stabilize the training of FFNs, particularly in the case of deeper models.
    out_features: number of output feature maps
    enumerate_layers: If true, layer names will be prefixed with their number
      within the model. This parameter affects only the way how model params are
      names, not the behavior.
    kernel_shape: The 3d shape of the convolution kernel
    native_input_size: The native spatial size of the model input. The model may
      be able to process input of different size, but the configured input is
      usually expected to work the best. Changing this parameter does not affect
      the inference.
  """

  features: int | Iterable[int] = 32
  depth: int = 12  # number of residual modules
  padding: str = 'same'
  dim: int = 3
  num_convs: int = 2
  use_layernorm: bool = True
  out_features: int = 1
  enumerate_layers: bool = False
  kernel_shape: tuple[int, int, int] = (3, 3, 3)
  native_input_size: tuple[int, int, int] | None = None


class ResConvStack(nn.Module):
  """Residual convstack."""

  config: ConvstackConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies the convstack to the input.

    Args:
      x: [batch, z, y, x, channels]-shaped input.

    Returns:
      convstack output
    """
    cfg = self.config

    layer_naming = param_util.LayerNaming(self.config.enumerate_layers)

    if isinstance(cfg.features, int):
      features = itertools.repeat(cfg.features)
    else:
      features = iter(cfg.features)

    labels = 'abcdefghijklmnopqrstuvwxyz'

    x = nn.Conv(
        next(features),
        self.config.kernel_shape[: cfg.dim],
        padding=cfg.padding,
        name=layer_naming.get_name('pre_a'),
    )(x)
    if cfg.use_layernorm:
      x = nn.LayerNorm()(x)
    for i in range(1, cfg.num_convs):
      x = nn.relu(x)
      x = nn.Conv(
          next(features),
          self.config.kernel_shape[: cfg.dim],
          padding=cfg.padding,
          name=layer_naming.get_name(f'pre_{labels[i]}'),
      )(x)

    for i in range(cfg.depth):
      mod_input = x
      if cfg.use_layernorm:
        x = nn.LayerNorm()(x)
      for j in range(0, cfg.num_convs):
        x = nn.relu(x)
        x = nn.Conv(
            next(features),
            self.config.kernel_shape[: cfg.dim],
            padding=cfg.padding,
            name=layer_naming.get_name(f'res{i}{labels[j]}'),
        )(x)

      if x.shape != mod_input.shape:
        crop_shape_zyx = x.shape[1 : 1 + cfg.dim]
        x += util.center_crop(mod_input, crop_shape_zyx)
      else:
        x += mod_input

    if cfg.use_layernorm:
      x = nn.LayerNorm()(x)
    x = nn.relu(x)
    return nn.Conv(
        cfg.out_features,
        (1, 1, 1)[: cfg.dim],
        name=layer_naming.get_name('output'),
    )(x)

  def compute_output_box_from_input_box(
      self, input_box: BoundingBox
  ) -> BoundingBox:
    """Computes the bounding box in the output volume.

    Args:
      input_box: The bounding box at the input of the model.

    Returns:
      The bounding box in the output volume.
    """
    normalized_padding = self.config.padding.lower()
    kernel_shape = self.config.kernel_shape
    if normalized_padding == 'valid':
      # Each layer contract by the (kernel shape - 1) / 2 voxels.
      # Each res block contains a number of convs + a skip connection. Only conv
      # layers contract.
      single_conv_contraction = (
          jnp.asarray(kernel_shape) - jnp.asarray((1, 1, 1))
      ) / 2
      num_contractions = self.config.num_convs * (self.config.depth + 1)
      return BoundingBox(
          input_box.start + num_contractions * single_conv_contraction,
          input_box.size - 2 * num_contractions * single_conv_contraction,
      )

    # When padding, the output of the model results in the same location.
    return input_box

  def compute_input_box_from_output_box(
      self, output_box: BoundingBox
  ) -> BoundingBox:
    """Computes the bounding box in the input volume.

    Args:
      output_box: The bounding box which should be inferred.

    Returns:
      The bounding box in the input volume.
    """

    normalized_padding = self.config.padding.lower()
    kernel_shape = self.config.kernel_shape
    if normalized_padding == 'valid':
      # Each layer contract by the (kernel shape - 1) / 2 voxels.
      # Each res block contains a number of convs + a skip connection. Only conv
      # layers contract.
      single_conv_contraction = (
          jnp.asarray(kernel_shape) - jnp.asarray((1, 1, 1))
      ) / 2
      num_contractions = self.config.num_convs * (self.config.depth + 1)
      return BoundingBox(
          output_box.start - num_contractions * single_conv_contraction,
          output_box.size + 2 * num_contractions * single_conv_contraction,
      )

    # When padding, the output of the model results in the same location.
    return output_box

  def get_bounding_box_calculator(self) -> 'ResConvStack':
    """Returns the bounding box calculator.

    Returns:
      The object capable of transforming bounding boxes between the input and
      the output volumes.
    """
    return self

  def get_native_output_size(self) -> tuple[int, int, int] | None:
    if not self.config.native_input_size:
      return None
    input_bounding_box = BoundingBox(
        start=(0, 0, 0), size=self.config.native_input_size
    )
    bbox_calculator = self.get_bounding_box_calculator()
    output_box = bbox_calculator.compute_output_box_from_input_box(
        input_bounding_box
    )
    return output_box.size

  def get_native_input_size(self) -> tuple[int, int, int] | None:
    return self.config.native_input_size


class ResConvNeXtStack(nn.Module):
  """Inspired by ConvNeXt: https://arxiv.org/abs/2201.03545."""

  config: ConvstackConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies the convstack to the input.

    Args:
      x: [batch, z, y, x, channels]-shaped input.

    Returns:
      convstack output
    """
    cfg = self.config

    if isinstance(cfg.features, int):
      features = itertools.repeat(cfg.features)
    else:
      features = iter(cfg.features)

    point_kernel = (1, 1, 1)[: cfg.dim]
    space_kernel = (7, 7, 7)[: cfg.dim]

    feat_out = next(features)
    x = nn.Conv(feat_out, space_kernel, padding=cfg.padding, name='pre_a')(x)
    x = nn.LayerNorm()(x)
    x = nn.Conv(feat_out * 4, point_kernel, padding=cfg.padding, name='pre_b')(
        x
    )
    x = nn.relu(x)
    x = nn.Conv(feat_out, point_kernel, padding=cfg.padding, name='pre_c')(x)

    for i in range(cfg.depth):
      mod_input = x
      feat_in, feat_out = feat_out, next(features)
      x = nn.Conv(
          feat_out,
          space_kernel,
          padding=cfg.padding,
          feature_group_count=feat_in,
          name=f'res{i}_a',
      )(x)
      x = nn.LayerNorm()(x)
      x = nn.Conv(
          feat_out * 4, point_kernel, padding=cfg.padding, name=f'res{i}_b'
      )(x)
      x = nn.relu(x)
      x = nn.Conv(
          feat_out, point_kernel, padding=cfg.padding, name=f'res{i}_c'
      )(x)
      if x.shape != mod_input.shape:
        crop_shape_zyx = x.shape[1 : 1 + cfg.dim]
        x += util.center_crop(mod_input, crop_shape_zyx)
      else:
        x += mod_input

    x = nn.relu(x)
    return nn.Conv(cfg.out_features, point_kernel, name='output')(x)

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
"""Utilities for JAX / FLAX models.

This file gets imported by XM launch scipts. Please keep dependencies
minimal and ensure that they work with the binary_import mechanism.
"""

from typing import Sequence

from connectomics.common import bounding_box
import jax
import jax.numpy as jnp


def center_crop_in_all_dimensions(
    x: jax.Array, expected_shape: Sequence[int]
) -> jax.Array:
  """Extracts a crop of the expected shape from the center of a tensor.

  No distinction is made between spatial, batch and channel dimensions.
  If the expected output size is larger than the input size, no cropping will be
  performed in the corresponding dimension.

  Args:
    x: The tensor which should be cropped.
    expected_shape: A sequence of dimensions after cropping.

  Returns:
    The cropped tensor.
  """
  starts = [max(0, (x - e) // 2) for x, e in zip(x.shape, expected_shape)]
  ends = [min(lim, s + e) for s, e, lim in zip(starts, expected_shape, x.shape)]
  slices = tuple([slice(s, e) for s, e in zip(starts, ends)])
  return x[slices]


def center_crop(x: jax.Array, crop_spatial_shape: Sequence[int]) -> jax.Array:
  """Extracts crop_shape from the center of a xZYXx tensor or xYXx tensor.

  Args:
    x: The tensor which should be cropped.
    crop_spatial_shape: The spatial shape after croppping.

  Returns:
    The cropped tensor.
  """
  return center_crop_in_all_dimensions(
      x,
      tuple(x.shape[: -(len(crop_spatial_shape) + 1)])
      + tuple(crop_spatial_shape)
      + (x.shape[-1],),
  )


def pad_symmetrically_in_all_dimensions(
    x: jax.Array, expected_shape: Sequence[int]
) -> jax.Array:
  """Symmetrically pads the provided tensor in all dimensions.

  No distinction is made between spatial, batch and channel dimensions.
  If the expected output size is smaller than the input size, no padding will be
  performed in the corresponding dimension.

  Args:
    x: The tensor which should be padded.
    expected_shape: A sequence of dimensions after padding.

  Returns:
    The cropped tensor.
  """
  total_padding = jnp.asarray(expected_shape) - jnp.asarray(x.shape)
  total_padding = jnp.clip(total_padding, 0)
  left_padding = total_padding // 2
  right_padding = total_padding - left_padding
  requested_padding = jnp.concatenate(
      (left_padding.reshape((-1, 1)), right_padding.reshape((-1, 1))), axis=1
  )
  return jnp.pad(x, requested_padding)


def pad_symmetrically(
    x: jax.Array, padded_spatial_shape: Sequence[int]
) -> jax.Array:
  """Spatially pads tensors with batch and channel dimensions.

  Batch and channel dimensions are not affected by the padding.
  Example correct input tensor shapes: xZYXc, xYXx.

  Args:
    x: The tensor which should be padded.
    padded_spatial_shape: The spatial shape after padding.

  Returns:
    The cropped tensor.
  """
  return pad_symmetrically_in_all_dimensions(
      x, (1,) + tuple(padded_spatial_shape) + (1,)
  )


def center_crop_bounding_box(
    original_box: bounding_box.BoundingBox, final_size_zyx: Sequence[int]
) -> bounding_box.BoundingBox:
  """Updates the bounding box to match the final tensor spatial size.

   Cropping assumes that the output tensor (of size `final_size`) corresponds to
   the center of the original bounding box.

  Args:
    original_box: The bounding box to be cropped.
    final_size_zyx: The tensor size after cropping.

  Returns:
    The cropped bounding box.
  """
  final_size_xyz = tuple(reversed(final_size_zyx))
  cropping_offsets = (original_box.size - final_size_xyz) // 2
  new_start = jnp.asarray(original_box.start) + cropping_offsets
  return bounding_box.BoundingBox(
      start=tuple(new_start), size=tuple(final_size_xyz)
  )

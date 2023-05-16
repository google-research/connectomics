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
"""Data shaping module supporting data.py."""

from typing import Sequence, Any
import tensorflow.compat.v2 as tf

_TPU_DTYPES = (tf.float32, tf.int32, tf.complex64, tf.int64, tf.bool,
               tf.bfloat16, tf.uint32)


def build_key_data(features: dict[str, Any],
                   data_kinds: Sequence[str],
                   box_size: Sequence[int],
                   num_pos_examples: int,
                   is_training: bool,
                   debug: bool = False,
                   use_tpu: bool = False) -> dict[str, Any]:
  """Assembles data product.

  Args:
    features: features dict as eg. loaded from tf.Example
    data_kinds: defines data product. Must be one of ['seg', 'em', 'masked_em']
    box_size: shape of patch (x, y, z)
    num_pos_examples: number of corresponding patches for the same entity
    is_training: are we training?
    debug: If True, will print out shape information of Tensors
    use_tpu: Whether execution will happen on a TPU. If True, data types that
      are incompatible with execution on the TPU will be deleted from the
      features dict, most notably strings.

  Returns:
   features: transformed dictionary. All prior data entries have been removed
     and the data product is stored under 'data'
  """
  data_shape = [
      box_size[0], box_size[1], box_size[2], num_pos_examples * len(data_kinds)
  ]

  data_channels = []
  for channel_id, data_kind in enumerate(data_kinds):
    channel_k = f"channel_{channel_id}"

    if data_kind == "em":
      data_channels.append(features[channel_k]["em"][:, 0])
    elif data_kind == "seg":
      data_channels.append(
          tf.cast(features[channel_k]["mask"][:, 0], dtype=tf.float32))
    elif data_kind == "masked_em":
      data_channels.append(
          tf.where(
              condition=features[channel_k]["mask"][:, 0],
              x=features[channel_k]["em"][:, 0],
              y=tf.zeros_like(features[channel_k]["em"][:, 0])))
    else:
      raise ValueError("data_kind must be one of ['em', 'seg', 'masked_em']")

    if not debug:
      del features[channel_k]

  # Reshape so that the pairs are combined with channels
  features["data"] = tf.reshape(
      tf.transpose(tf.concat(data_channels, axis=-1), [1, 2, 3, 0, 4]),
      data_shape)

  features["data"] = tf.cast(features["data"], dtype=tf.float32)

  if use_tpu and not debug:
    for key in list(features.keys()):
      if features[key].dtype not in _TPU_DTYPES:
        del features[key]

  if is_training and not debug:
    for key in list(features.keys()):
      if key not in ["data", "distance"]:
        del features[key]

  return features


def pad_to_batch(features: dict[str, Any], batch_size: int) -> dict[str, Any]:
  """Pads incomplete batch due to end of dataset to correct size.

  Only relevant during inference.

  Args:
    features: features dict as eg. loaded from tf.Example. The "data" entry is
      padded to the batch size.
    batch_size: batch size

  Returns:
    features: transformed dictionary
  """
  batch_diff = batch_size - tf.shape(features["data"])[0]
  paddings = [[0, batch_diff]] + [[0, 0]] * (features["data"].shape.ndims - 1)
  features["data"] = tf.pad(
      features["data"], paddings=paddings, mode="constant")

  return features

# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Point Infinity model, see https://arxiv.org/pdf/2404.03566."""

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp

from connectomics.jax import spatial
from connectomics.jax.models import point
from connectomics.mogen import reorder
from connectomics.mogen import utils
from connectomics.mogen.models import minimal_network_utils
from connectomics.mogen.models import minimal_vit as vit


@struct.dataclass
class PointInfinityConfig:
  """Configuration for the Point Infinity model."""

  point_dim: int = 256
  latent_dim: int = 512
  n_latents: int = 256
  n_blocks: int = 6
  n_subblocks: int = 4
  n_heads: int = 8
  k_nn: int = 16
  dropout: float = 0.0
  combine_z: int = 1
  n_combine_samples: int = 1
  out_dim: int | None = None


class PointInfinityBlock(nn.Module):
  """Point Infinity block."""

  config: PointInfinityConfig

  @nn.compact
  def __call__(
      self,
      x,
      z,
  ):
    # TODO(riegerfr): ablate norms + their initialization
    z = z + vit.MultiHeadDotProductAttention(
        num_heads=self.config.n_heads,
        normalize_qk=True,
    )(
        nn.RMSNorm()(z),
        nn.RMSNorm()(x),
    )
    z = z + (
        vit.MlpBlock(
            dropout=self.config.dropout,
        )(nn.RMSNorm()(z), None, True)
    )

    for _ in range(self.config.n_subblocks):
      z_normed = nn.RMSNorm()(z)
      z = z + vit.MultiHeadDotProductAttention(
          num_heads=self.config.n_heads,
          normalize_qk=True,
      )(
          z_normed,
          z_normed,
      )
      z = z + (
          vit.MlpBlock(dropout=self.config.dropout)(nn.RMSNorm()(z), None, True)
      )

    x = x + vit.MultiHeadDotProductAttention(
        num_heads=self.config.n_heads,
        normalize_qk=True,
    )(
        nn.RMSNorm()(x),
        nn.RMSNorm()(z),
    )
    x = x + (
        vit.MlpBlock(dropout=self.config.dropout)(nn.RMSNorm()(x), None, True)
    )
    return x, z


class PointInfinity(nn.Module):
  """Point Infinity model."""

  config: PointInfinityConfig

  @nn.compact
  def __call__(
      self,
      coord: jax.Array,
      feat: jax.Array | None = None,
      t: jax.Array | None = None,
      cond: jax.Array | None = None,
      deterministic: bool = True,
      point_cond_mask: jax.Array | None = None,
  ) -> jax.Array:
    if t is not None:
      t_emb = vit.MlpBlock(dropout=self.config.dropout)(
          minimal_network_utils.get_timestep_embedding(
              t,
              embedding_dim=self.config.latent_dim,
              max_time=1.0,
          )[:, None, :],
          None,
          True,
      )[:, 0, :]
    else:
      t_emb = None
    cond = nn.Dense(self.config.latent_dim)(cond) if cond is not None else None

    if self.config.combine_z > 1:
      if feat is not None:
        raise NotImplementedError(
            'Features and combine_z > 1 not implemented yet.'
        )
      coord, reverse_indices = reorder.reorder_z_sfc_with_reverse(coord)
      if self.config.n_combine_samples > 1:
        raise NotImplementedError('Combine mask not implemented yet.')
    else:
      reverse_indices = None

    x = coord
    if self.config.n_combine_samples > 1:
      if feat is not None:
        raise NotImplementedError(
            'Features and combine_samples > 1 not implemented yet.'
        )
      combine_map = utils.get_combine_map(
          self.config.n_combine_samples, coord.shape[1]
      )
      combine_map = combine_map[..., None]
      combine_map = jnp.repeat(combine_map, x.shape[0], axis=0)
      x = jnp.concatenate((x, combine_map), axis=-1)

    if point_cond_mask is not None:
      assert self.config.n_combine_samples == 1
      assert point_cond_mask.shape[1] == coord.shape[1]
      assert point_cond_mask.shape[0] == coord.shape[0]
      assert self.config.combine_z == 1
      x = jnp.concatenate((x, point_cond_mask[:, :, None]), axis=-1)

    if self.config.k_nn > 0:
      if (
          coord.shape[1] > 8192
      ):  # TODO(riegerfr): Consider a common wrapper function for dispatching

        _, k_nn_idx = spatial.kdnn(coord, self.config.k_nn + 1)
        k_nn_idx = k_nn_idx[..., 1:]  # Remove the point itself
      else:
        k_nn_idx = spatial.knn(coord, coord, self.config.k_nn)
      k_nn_coord = (
          point.batch_lookup(coord, k_nn_idx) - coord[:, :, None]
      ).reshape(coord.shape[0], coord.shape[1], -1)
      x = jnp.concatenate((x, k_nn_coord), axis=-1)

    if self.config.combine_z > 1:
      x = x.reshape(
          x.shape[0],
          x.shape[1] // self.config.combine_z,
          x.shape[2] * self.config.combine_z,
      )

    if feat is not None:
      x = jnp.concatenate((x, feat), axis=-1)

    x = nn.Dense(self.config.point_dim)(x)

    z = jnp.repeat(
        nn.Embed(
            self.config.n_latents, self.config.latent_dim
        )(  # TODO(riegerfr): scale init
            jnp.arange(self.config.n_latents)
        )[
            None
        ],
        coord.shape[0],
        axis=0,
    )
    z = jnp.concat((z, t_emb[:, None, :]), axis=1) if t_emb is not None else z
    z = jnp.concat((z, cond[:, None, :]), axis=1) if cond is not None else z

    for _ in range(self.config.n_blocks):
      x, z = PointInfinityBlock(self.config)(x, z)

    x = nn.Dense(
        (
            (coord.shape[-1] + (feat.shape[-1] if feat is not None else 0))
            if self.config.out_dim is None
            else self.config.out_dim
        )
        * self.config.combine_z,
    )(x)

    if self.config.combine_z > 1:
      x = x.reshape(x.shape[0], coord.shape[1], -1)
      x = x[
          jnp.arange(x.shape[0])[:, None],
          reverse_indices,
      ]

    return x

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
"""Networks for processing point sets.

Relevant papers:
  PCT: https://arxiv.org/pdf/2012.09164.pdf
  PointMLP: https://arxiv.org/pdf/2202.07123.pdf
  PointTransformer:
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf
  PointNeXt: https://arxiv.org/pdf/2206.04670.pdf

Role of non-parametric operations:
  https://arxiv.org/pdf/2303.08134.pdf
"""

from collections.abc import Sequence

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from connectomics.jax import spatial


class EmbedPos(nn.Module):
  """Position embedding from XYZ coordinates (differences)."""

  features: int = 32

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    """Computes position embedding from XYZ coordinates (differences).

    Args:
      x: [...n, 3]: XYZ coordinate (difference)

    Returns:
      [...n, features] embedding
    """
    x = nn.Dense(3)(x)
    # TODO(riegerfr, mjanusz): verify training instability for higher dim than 3
    x = nn.relu(x)
    x = nn.Dense(self.features)(x)
    return x


def batch_lookup(x: jax.Array, idx: jax.Array) -> jax.Array:
  """Performs a batch lookup.

  Args:
    x: [b, n, c] array with data
    idx: [b, m, c] array with indices to look up

  Returns:
    [b, n, m, c] array
  """
  b = x.shape[0]
  batch = jnp.repeat(jnp.arange(b), np.prod(idx.shape[-2:])).reshape(idx.shape)
  return x[batch, idx, :]


def _apply_pool(
    x: jax.Array, pool_type: str, mask: jax.Array | None = None
) -> jax.Array:
  """Applies a pooling operation over the point dimension.

  Args:
    x: [..., k, c] features to pool over the k dimension
    pool_type: pooling type ('max', 'mean', 'lP', 'attention')
    mask: [..., k] boolean mask, True for valid entries. If None, all entries
      are valid.

  Returns:
    [..., c] pooled features
  """
  if mask is not None:
    mask_expanded = mask[..., None]

  if pool_type == 'max':
    if mask is not None:
      x = jnp.where(mask_expanded, x, -jnp.inf)
    return jnp.max(x, axis=-2)
  elif pool_type.startswith('l'):
    p = float(pool_type[1:])
    if mask is not None:
      x = jnp.where(mask_expanded, x, 0.0)
      count = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)
      return (jnp.sum(jnp.abs(x) ** p, axis=-2) / count + 1e-8) ** (1.0 / p)
    return (jnp.mean(jnp.abs(x) ** p, axis=-2) + 1e-8) ** (1.0 / p)
  elif pool_type == 'attention':
    attn = nn.Dense(1)(x)
    if mask is not None:
      attn = jnp.where(mask_expanded, attn, -jnp.inf)
    attn = nn.softmax(attn, axis=-2)
    return jnp.sum(x * attn, axis=-2)
  else:
    if mask is not None:
      x = jnp.where(mask_expanded, x, 0.0)
      count = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)
      return jnp.sum(x, axis=-2) / count
    return jnp.mean(x, axis=-2)


class ConditionedLayerNorm(nn.Module):
  """Layer normalization with optional conditioning."""

  @nn.compact
  def __call__(self, x: jax.Array, cond: jax.Array | None = None) -> jax.Array:
    """Applies layer normalization, optionally conditioned on `cond`.

    Args:
      x: [b,..., c] input data
      cond: [b, c'] conditioning vector

    Returns:
      [b,..., c] normalized data
    """
    if cond is None:
      return nn.LayerNorm()(x)
    else:
      x = nn.LayerNorm(use_bias=False, use_scale=False)(x)

      # Predict bias/scale from `cond`
      bias_scale = nn.Dense(2 * x.shape[-1])(cond)
      bias_scale = jnp.expand_dims(
          bias_scale,
          axis=[i + 1 for i in range(len(x.shape) - len(bias_scale.shape))],
      )
      bias, scale = jnp.split(bias_scale, 2, axis=-1)

      # For training stability, add 1.0 s.t. the initial scale is not (expected)
      # 0.0 but 1.0 as for vanila LayerNorm.
      return x * (scale + 1.0) + bias


def _get_norm(name: str):
  if name.startswith('layer'):
    return ConditionedLayerNorm
  else:
    return lambda: lambda x, *args, **kwargs: x


@struct.dataclass
class GroupMLPConfig:
  """Config for a redisual MLP for point groups.

  Attributes:
    num_layers: number of layers
    features: number of output features
    pool_type: 'max' or 'mean' pooling
    normalizer: type of normalizer to apply before the nonlinearity
  """

  num_layers: int
  features: int
  pool_type: str
  normalizer: str


class GroupMLP(nn.Module):
  """Residual MLP for groups of points."""

  config: GroupMLPConfig

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      cond: jax.Array | None = None,
      mask: jax.Array | None = None,
  ) -> jax.Array:
    """Applies a residual MLP to every group of up to k neighbors.

    Summarizes the result with pooling.

    Args:
      x: [b, n, k, c] point group features; each of 'n' points is associated
        with 'k' neighbors
      cond: [b, c'] conditioning vector
      mask: [b, n, k] boolean mask, True for valid neighbors

    Returns:
      [b, n, feat] processed data
    """
    cfg = self.config
    norm = _get_norm(cfg.normalizer)

    for _ in range(cfg.num_layers - 1):
      x = nn.Dense(cfg.features // 2)(x)
      x = norm()(x, cond=cond)
      x = nn.relu(x)

    x = nn.Dense(cfg.features)(x)
    x = norm()(x, cond=cond)
    x = nn.relu(x)

    x = _apply_pool(x, cfg.pool_type, mask=mask)

    return x


@struct.dataclass
class LocalAggregationConfig:
  """Config for the local aggregation operation.

  Attributes:
    mlp_config: settings for the MLP to apply to evert group
    k: number of nearest neighbors for the kNN op
    skip_conn: whether to include a skip connection from anchor points
    anchor_concat: whether to concatenate anchor features to those of every
      neighbor point
    anchor_feat_offset: whether to subtract anchor features from the neighbor
      features
    radius: if set, use ball query instead of kNN; only neighbors within this
      distance are aggregated
    max_k: maximum number of neighbors for ball query; required when radius
      is set
  """

  mlp_config: GroupMLPConfig
  k: int = 32
  skip_conn: bool = False
  anchor_concat: bool = False
  anchor_feat_offset: bool = True
  radius: float | None = None
  max_k: int | None = None


class LocalAggregation(nn.Module):
  """Forms point groups around anchor points.

  Every group is processed in parallel by an MLP, the results
  of which are pooled to compute the updated features of the
  anchor points.
  """

  config: LocalAggregationConfig

  @nn.compact
  def __call__(
      self,
      feat: jax.Array,
      coord: jax.Array,
      anchor_feat: jax.Array,
      anchor_coord: jax.Array,
      cond: jax.Array | None = None,
  ) -> jax.Array:
    """Applies the local aggregation op.

    Args:
      feat: [b, n, c] features of all points
      coord: [b, n, 3] coordinates of all points
      anchor_feat: [b, m, c] features of anchor points
      anchor_coord: [b, m, 3] coordinates of anchor points
      cond: [b, c'] per sample conditioning vector

    Returns:
      [b, m, c] new features of anchor points
    """
    cfg = self.config

    # TODO(mjanusz): try topology-aware subsampling instead of kNN

    # Find up to k neighbors for every anchor point.
    mask = None
    if cfg.radius is not None:
      assert cfg.max_k is not None
      k_idx, mask = spatial.ball_query(
          anchor_coord, coord, cfg.radius, cfg.max_k
      )
      num_k = cfg.max_k
    else:
      k_idx = spatial.KNN(cfg.k)(anchor_coord, coord)  # [b, n, k]
      num_k = cfg.k

    k_feat = batch_lookup(feat, k_idx)  # [b, n, k, c]
    if cfg.anchor_feat_offset:
      k_feat -= anchor_feat[..., None, :]
      # TODO(riegerfr, mjanusz): k_feat[:,:,0,:] is all 0, inefficient,
      # exclude anchor in KNN?

    # Add normalized distances to anchor as additional features.
    k_coord = batch_lookup(coord, k_idx)  # [b, n, k, 3]
    delta_coord = k_coord - anchor_coord[..., None, :]
    if cfg.radius is not None:
      delta_coord = delta_coord / cfg.radius
    else:
      delta_coord /= jnp.max(
          jnp.linalg.norm(delta_coord, axis=-1, keepdims=True),
          axis=(-1, -2, -3),
          keepdims=True,
      )
    new_feat = jnp.concatenate(
        [delta_coord, k_feat], axis=-1
    )  # [b, n, k, c + 3]

    if cfg.anchor_concat:
      # Concat the anchor (fps) features to those of every neighbor.
      new_feat = jnp.concatenate(
          [new_feat, jnp.repeat(anchor_feat[..., None, :], num_k, axis=-2)],
          axis=-1,
      )

    if mask is not None:
      new_feat = jnp.where(mask[..., None], new_feat, 0.0)

    new_feat = GroupMLP(cfg.mlp_config)(
        new_feat, cond=cond, mask=mask
    )  # [b, n, c]

    if cfg.skip_conn:
      # Skip connection from the anchor points.
      ident = nn.Dense(new_feat.shape[-1])(anchor_feat)
      new_feat = nn.relu(new_feat + ident)

    return new_feat


class SetAbstraction(nn.Module):
  """Downsamples the point set to a target size.

  Attributes:
    num_points: target number of points to downsample to
    random_seed_node: whether to use random seed nodes
    config: settings for local aggregation
    use_random_subsample: if True, use random subsampling instead of FPS
  """

  num_points: int
  random_seed_node: bool
  config: LocalAggregationConfig
  use_random_subsample: bool = False

  @nn.compact
  def __call__(
      self,
      feat: jax.Array,
      coord: jax.Array,
      cond: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    # Select anchor points via subsampling (only when actually downsampling).
    if self.num_points >= coord.shape[-2]:
      sub_coord = coord
      sub_feat = feat
    elif self.use_random_subsample:
      rng = self.make_rng('subsample')
      sub_coord, sub_idx = spatial.random_subsample_points(
          coord, self.num_points, key=rng
      )
      sub_feat = batch_lookup(feat, sub_idx[..., None])[..., 0, :]
    else:
      sub_coord, sub_idx = spatial.FPS(
          self.num_points, self.random_seed_node
      )(coord)
      sub_feat = batch_lookup(feat, sub_idx[..., None])[..., 0, :]

    new_feat = LocalAggregation(self.config)(
        feat, coord, sub_feat, sub_coord, cond=cond
    )
    return new_feat, sub_coord


@struct.dataclass
class InvResMLPConfig:
  la_config: LocalAggregationConfig
  features: Sequence[int]
  normalizer: str


class InvResMLP(nn.Module):
  """Aggregates data and applies a residual MLP."""

  config: InvResMLPConfig

  @nn.compact
  def __call__(
      self,
      feat: jax.Array,
      coord: jax.Array,
      cond: jax.Array | None = None,
  ) -> jax.Array:
    cfg = self.config

    norm = _get_norm(cfg.normalizer)
    x = LocalAggregation(cfg.la_config)(feat, coord, feat, coord, cond=cond)

    for f in cfg.features:
      x = nn.Dense(f)(x)
      x = norm()(x, cond=cond)
      x = nn.relu(x)

    if feat.shape[-1] != x.shape[-1]:
      feat = nn.Dense(x.shape[-1])(feat)

    x = x + feat
    x = nn.relu(x)
    return x


@struct.dataclass
class PointNeXtStageConfig:
  """Configs for a single stage of a PointNeXt network."""

  blocks: Sequence[InvResMLPConfig]
  sa_config: LocalAggregationConfig | None = None
  random_seed_node: bool = False
  downsample: int = 0
  use_random_subsample: bool = False


class PointNeXtStage(nn.Module):
  """Single stage of a PointNeXt network."""

  config: PointNeXtStageConfig

  @nn.compact
  def __call__(
      self,
      feat: jax.Array,
      coord: jax.Array,
      cond: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    cfg = self.config

    if cfg.sa_config is not None:
      feat, coord = SetAbstraction(
          feat.shape[-2] // cfg.downsample,
          cfg.random_seed_node,
          cfg.sa_config,
          use_random_subsample=cfg.use_random_subsample,
      )(feat, coord, cond=cond)
    else:
      # TODO(mjanusz): Consider local per point processing here.
      # Typically used at the beginning of the network.
      pass

    for blk_cfg in cfg.blocks:
      feat = InvResMLP(blk_cfg)(feat, coord, cond=cond)

    return feat, coord


@struct.dataclass
class PointNeXtEncoderConfig:
  stages: Sequence[PointNeXtStageConfig]
  embed_dim: int
  normalizer: str


class PointNeXtEncoder(nn.Module):
  """Encoder of PointNeXt."""

  config: PointNeXtEncoderConfig

  @nn.compact
  def __call__(
      self,
      feat: jax.Array,
      coord: jax.Array,
      return_feat_coord_tuples: bool = False,
      cond: jax.Array | None = None,
  ) -> jax.Array | Sequence[tuple[jax.Array, jax.Array]]:
    cfg = self.config

    norm = _get_norm(cfg.normalizer)
    feat = nn.Dense(cfg.embed_dim)(feat)
    feat = norm()(feat, cond=cond)
    feat = nn.relu(feat)
    feat = feat + EmbedPos(cfg.embed_dim)(coord)

    feat_coord_tuples = [(feat, coord)]
    for stage_cfg in cfg.stages:
      feat, coord = PointNeXtStage(stage_cfg)(feat, coord, cond=cond)
      feat_coord_tuples.append((feat, coord))
    if return_feat_coord_tuples:
      return feat_coord_tuples
    else:
      return feat


@struct.dataclass
class PointNeXtClassifierConfig:
  """Configuration settings for a PointNeXt classifier.

  Attributes:
    enc_config: encoder setitngs
    pool_type: pooling op to use (max, mean) for global pooling
    normalizer: normalizer to use (layer, '')
    features: sizes of the hidden layers of the classifier head
    num_classes: number of classes for the classifier
    pca_align: whether to align the point cloud using PCA
    num_vector_feats: number of vector features, stored in the first
      3 * num_vector_feats elements of feats
    use_random_subsample: whether to use random subsampling instead of FPS
  """

  enc_config: PointNeXtEncoderConfig
  pool_type: str
  normalizer: str
  features: Sequence[int]
  num_classes: int
  pca_align: bool
  num_vector_feats: int
  use_random_subsample: bool = False


class PointNeXtClassifier(nn.Module):
  """Classifier on top of PointNeXt encoder."""

  config: PointNeXtClassifierConfig

  @nn.compact
  def __call__(
      self, feat: jax.Array | None, coord: jax.Array, ret_pooled: bool = False
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    cfg = self.config
    if feat is None:
      feat = jnp.zeros((coord.shape[0], coord.shape[1], 1))

    if cfg.pca_align:
      coord, feat = spatial.pca_align(coord, feat, cfg.num_vector_feats)

    x = PointNeXtEncoder(cfg.enc_config)(feat, coord)

    x = _apply_pool(x, cfg.pool_type)
    x_pooled = x

    norm = _get_norm(cfg.normalizer)
    # Classification head.
    for num_features in cfg.features:
      x = norm()(x)
      x = nn.Dense(num_features)(x)
      x = nn.relu(x)

    # Final projection.
    x = nn.Dense(cfg.num_classes)(x)
    return x if not ret_pooled else (x_pooled, x)

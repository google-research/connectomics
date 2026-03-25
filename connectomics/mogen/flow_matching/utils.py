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
"""Utilities for Flow Matching training."""

import base64
from collections.abc import Callable
import functools
import math
import shutil
import tempfile
from typing import Any, Type

from absl import logging
from connectomics.jax import spatial
from connectomics.jax.models import point
from connectomics.mogen import reorder
from connectomics.mogen import utils as mogen_utils
from connectomics.mogen.flow_matching import schedules
from connectomics.mogen.models import pointinfinity
from etils import epath
from ffn.inference import storage
import flax
import flax.linen as nn
from IPython import display
import jax
import jax.numpy as jnp
from matplotlib import animation
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import optax
import plotly.express as px
import plotly.graph_objs as go
from sklearn import decomposition
import tqdm

from cmmd import distance


N_PLOT = 16


@flax.struct.dataclass
class TrainState:
  """Training state for Flow Matching."""

  step: int
  params: flax.core.FrozenDict[str, Any]
  ema_params: flax.core.FrozenDict[str, Any]
  batch_stats: flax.core.FrozenDict[str, Any]
  opt_state: optax.OptState | None
  min_s_mmd_train: float | None = None


def get_model(
    model_rng: jax.Array,
    init_coord: jax.Array,
    init_feat: jax.Array | None,
    config: ml_collections.ConfigDict,
    cond: jax.Array | None = None,
    point_cond_mask: jax.Array | None = None,
    n_combine_samples: int = 1,
    wrapper_cls: Type[nn.Module] | None = None,
    wrapper_kwargs: dict[str, Any] | None = None,
) -> tuple[
    nn.Module,
    flax.core.FrozenDict[str, Any],
    flax.core.FrozenDict[str, Any] | None,
]:
  """Creates and initializes the model.

  Args:
    model_rng: PRNG key for model initialization.
    init_coord: Initial coordinates for model input.
    init_feat: Initial features for model input.
    config: Configuration dictionary.
    cond: Initial conditioning for model input.
    point_cond_mask: Mask for points to condition on.
    n_combine_samples: Number of samples to combine.
    wrapper_cls: Class to wrap the model.
    wrapper_kwargs: Keyword arguments for the wrapper class.

  Returns:
    Model, parameters, and batch statistics.
  """

  if config.model_type == 'pointinfinity':
    model = pointinfinity.PointInfinity(
        pointinfinity.PointInfinityConfig(
            point_dim=config.pfty_point_dim,
            latent_dim=config.pfty_latent_dim,
            n_latents=config.pfty_n_latents,
            n_blocks=config.pfty_n_blocks,
            n_subblocks=config.pfty_n_subblocks,
            n_heads=config.pfty_n_heads,
            k_nn=config.pfty_k_nn,
            dropout=config.pfty_dropout,
            combine_z=config.combine_z,
            n_combine_samples=n_combine_samples,
            out_dim=config.out_dim
            if hasattr(config, 'out_dim') and config.out_dim > 0
            else None,
        )
    )
  else:
    raise ValueError(f'Unknown model type: {config.model_type}')

  if wrapper_cls is not None:
    model = wrapper_cls(model=model, **(wrapper_kwargs or {}))

  variables = model.init(
      model_rng,
      coord=init_coord,
      feat=init_feat if config.use_feat else None,
      t=jnp.ones((init_coord.shape[0],))
      if not hasattr(config, 'use_time') or config.use_time
      else None,
      cond=cond,
      point_cond_mask=point_cond_mask,
  )
  params = variables['params']
  batch_stats = variables.get('batch_stats', None)
  return model, params, batch_stats


# Flow Matching following https://arxiv.org/pdf/2412.06264
def compute_loss(
    model: nn.Module,
    variables: flax.core.FrozenDict[str, Any] | dict[str, Any],
    rng: jax.Array,
    coord: jax.Array,
    feat: jax.Array,
    is_train: bool,
    schedule: str = 'linear',
    cond: jax.Array | None = None,
    point_cond: int = 0,
    do_ott: bool = False,
    reorder_type: str = 'none',
    reorder_noise_strength: float = 0.0,
    point_cond_sample_threshold: float = 0.0,
    feat_cond_dropout_threshold: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
  """Computes the flow matching loss.

  The loss is the MSE between the predicted and actual "velocity field".

  Args:
    model: Model to use.
    variables: Model variables.
    rng: PRNG key.
    coord: Input coordinates (shape: (batch_size, n_points, 3)).
    feat: Input features (shape: (batch_size, n_points, feat_dim)).
    is_train: Whether in training mode.
    schedule: Time schedule name (see schedules.py).
    cond: Conditioning for model input.
    point_cond: Number of points to condition on.
    do_ott: If True, use optimal transport based loss.
    reorder_type: How to reorder the point cloud. ('none', 'axes', 'ot')
    reorder_noise_strength: Strength of noise added after reordering.
    point_cond_sample_threshold: Probability to sample points for conditioning.
    feat_cond_dropout_threshold: Probability to drop features for conditioning.

  Returns:
    Loss value and auxiliary outputs.
  """
  rng_x, rng_t, rng_dropout, rng_coord_cond, rng_feat_cond, rng_re_noise = (
      jax.random.split(rng, 6)
  )
  x_1 = jnp.concatenate((coord, feat), axis=2) if feat is not None else coord
  x_0 = jax.random.normal(rng_x, x_1.shape)
  if reorder_type == 'axes':
    assert feat is None
    x_0 = reorder.reorder_z_sfc(x_0)
    x_1 = reorder.reorder_z_sfc(x_1)
    # adding noise after reordering helps:
    # https://openreview.net/pdf?id=62Ff8LDAJZ
    x_0 = (
        jnp.sqrt(reorder_noise_strength)
        * jax.random.normal(rng_re_noise, x_0.shape)
        + jnp.sqrt(1 - reorder_noise_strength) * x_0
    )
  elif reorder_type == 'ot':
    assert feat is None
    x_0, x_1 = reorder.vmap_ot(x_0, x_1)
  elif reorder_type == 'none':
    pass
  else:
    raise ValueError(f'Unknown reorder type: {reorder_type}')

  t = jax.random.uniform(rng_t, (x_1.shape[0],), minval=0.0, maxval=1.0)
  t = schedules.t_schedule(t, schedule)
  t_exp = jnp.expand_dims(t, range(1, len(x_1.shape)))
  x_t = (1 - t_exp) * x_0 + t_exp * x_1
  dx_t = x_1 - x_0

  if point_cond > 0:

    point_cond_mask = (
        jnp.zeros(x_1.shape[:2], dtype=bool).at[0, :point_cond].set(True)
    )
    sample_mask = (
        jax.random.uniform(rng_coord_cond, (x_1.shape[0],))
        < point_cond_sample_threshold
    )

    point_cond_mask = point_cond_mask * sample_mask[:, None]

    x_t = (
        x_t * (~point_cond_mask[:, :, None]) + x_1 * point_cond_mask[:, :, None]
    )
  else:
    point_cond_mask = None

  if cond is not None:
    rng1, rng2, rng3 = jax.random.split(
        rng_feat_cond,
        3,
    )
    # TODO(riegerfr): rename rng_feat_cond and other uses of feat to cond
    thresholds = jax.random.randint(
        rng1, (cond.shape[0], 1), 0, cond.shape[-1] + 1
    ).astype(cond.dtype) / float(cond.shape[-1])
    final_mask_f = (
        (jax.random.uniform(rng2, (cond.shape[0], cond.shape[-1])) < thresholds)
        * (
            jax.random.uniform(rng3, (cond.shape[0], 1))
            < feat_cond_dropout_threshold
        )
    ).astype(cond.dtype)
    cond = jnp.concatenate((cond * final_mask_f, final_mask_f), axis=-1)

  apply_out = model.apply(
      variables,
      coord=x_t[:, :, :3],
      feat=x_t[:, :, 3:] if feat is not None else None,
      t=t,
      rngs={'dropout': rng_dropout},
      mutable=is_train,
      deterministic=not is_train,
      cond=cond,
      point_cond_mask=point_cond_mask,
  )
  pred_dx_t, batch_stats = (
      (apply_out[0], apply_out[1].get('batch_stats', None))
      if is_train
      else (apply_out, None)
  )

  if do_ott:
    # TODO(riegerfr): minimize the corresponding path length
    # from x_t predict pred_x_1
    # coordinates are first 3 dims, then features
    pred_x_1 = x_t[:, :, :3] + (1 - t_exp) * pred_dx_t[:, :, :3]
    # compute squared dist of pred_x_1 and x_1
    dists = ((pred_x_1[:, :, None] - x_1[:, None, :]) ** 2).sum(-1)

    all_loss = ((dx_t[:, :, None] - pred_dx_t[:, None, :]) ** 2).sum(-1)

    # take closest (along last axis)
    min_dists_src_tgt = jnp.argmin(
        dists, axis=-1, keepdims=True
    )  # TODO(riegerfr): with softmax instead of argmin? (and temperature)
    # compute dist of dx_t and pred_dx_t for these closest
    loss_src_tgt = jnp.take_along_axis(
        all_loss, min_dists_src_tgt, axis=-1
    ).mean(axis=(-2, -1))

    # take closest (along second last axis)
    min_dists_tgt_src = jnp.argmin(dists, axis=-2, keepdims=True)
    # compute dist of dx_t and pred_dx_t for these closest
    loss_tgt_src = jnp.take_along_axis(
        all_loss, min_dists_tgt_src, axis=-2
    ).mean(axis=(-2, -1))

    elementwise_loss = loss_src_tgt + loss_tgt_src
  else:
    elementwise_loss = (dx_t - pred_dx_t) ** 2

    if point_cond_mask is not None:
      elementwise_loss = elementwise_loss * (~point_cond_mask[:, :, None])

    elementwise_loss = elementwise_loss.mean(axis=range(1, len(x_1.shape)))

  # TODO(riegerfr): extend loss:
  # v pred mse + (pred_1 - x_1)**2 + (pred_0-x_0)**2
  # TODO(riegerfr): penalize curvature directly? get second derivative?
  loss = elementwise_loss.mean()

  return loss, {
      't': t,
      'elementwise_loss': elementwise_loss,
      'loss': loss,
      'batch_stats': batch_stats,
  }


def update_state(
    model: nn.Module,
    state: TrainState,
    optimizer: optax.GradientTransformation,
    coord: jax.Array,
    feat: jax.Array,
    rng: jax.Array,
    polyak_decay: float,
    schedule: str = 'linear',
    cond: jax.Array | None = None,
    point_cond: int = 0,
    do_ott: bool = False,
    reorder_type: str = 'none',
    reorder_noise_strength: float = 0.0,
    point_cond_sample_threshold: float = 0.0,
    feat_cond_dropout_threshold: float = 0.0,
    comp_loss: Callable[..., Any] = compute_loss,
) -> tuple[TrainState, Any]:
  """Performs a single training step.

  Args:
    model: Model to use.
    state: Training state.
    optimizer: Optimizer to use.
    coord: Input coordinates (shape: (batch_size, n_points, 3)).
    feat: Input features (shape: (batch_size, n_points, feat_dim)).
    rng: PRNG key.
    polyak_decay: Polyak decay factor.
    schedule: Time schedule name.
    cond: Conditioning for model input.
    point_cond: Number of points to condition on.
    do_ott: Whether to use optimal transport.
    reorder_type: How to reorder the point cloud. ('none', 'axes', 'ot')
    reorder_noise_strength: Strength of noise added after reordering.
    point_cond_sample_threshold: Probability to sample points for conditioning.
    feat_cond_dropout_threshold: Probability to drop features for conditioning.
    comp_loss: Loss function to use.

  Returns:
    Updated training state and auxiliary outputs.
  """

  def loss_fn(params):
    variables = {'params': params}
    if state.batch_stats:
      variables['batch_stats'] = state.batch_stats
    return comp_loss(
        model,
        variables,
        rng,
        coord=coord,
        feat=feat,
        schedule=schedule,
        is_train=True,
        cond=cond,
        point_cond=point_cond,
        do_ott=do_ott,
        reorder_type=reorder_type,
        reorder_noise_strength=reorder_noise_strength,
        point_cond_sample_threshold=point_cond_sample_threshold,
        feat_cond_dropout_threshold=feat_cond_dropout_threshold,
    )

  grad_fn = jax.grad(loss_fn, has_aux=True)
  grad, aux = grad_fn(state.params)
  updates, new_opt_state = optimizer.update(grad, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)

  # EMA from
  # http://google3/third_party/py/scenic/projects/modified_simple_diffusion/trainer.py;l=248;rcl=615399109
  decay_warmup = (1.0 + state.step) / (10.0 + state.step)
  polyak_decay_updated = jnp.minimum(polyak_decay, decay_warmup)
  ema_params = jax.tree_util.tree_map(
      lambda old, new: polyak_decay_updated * old
      + (1 - polyak_decay_updated) * new,
      state.ema_params,
      new_params,
  )

  return (
      state.replace(
          step=state.step + 1,
          params=new_params,
          opt_state=new_opt_state,
          batch_stats=aux['batch_stats'],
          ema_params=ema_params,
      ),
      aux,
  )


def guided_apply(
    model: nn.Module,
    variables: flax.core.FrozenDict[str, Any] | dict[str, Any],
    x_t: jax.Array,
    t: jax.Array,
    guidance_scale: jax.Array | None = None,
    cond: jax.Array | None = None,
    point_cond_mask: jax.Array | None = None,
    guide: bool = False,
) -> tuple[Any, dict[str, Any]]:
  """Applies the model with optional guidance on a conditioning vector.

  Args:
    model: Model to use.
    variables: Model variables.
    x_t: Input data of shape (batch_size, n_points, 3).
    t: Time of shape (batch_size,).
    guidance_scale: Scale for guidance.
    cond: Conditioning for model input.
    point_cond_mask: Mask for points to condition on.
    guide: Whether to guide the model.

  Returns:
    Prediction of the model.
  """
  # TODO(riegerfr): split in cond vect guide and cond point guide
  # (also for weight/guidance scale)
  pred = model.apply(
      variables,
      coord=x_t[:, :, :3],
      feat=x_t[:, :, 3:] if x_t.shape[2] > 3 else None,
      t=t,
      mutable=False,
      cond=cond,
      point_cond_mask=point_cond_mask,
  )
  if guide:
    pred_no_cond = model.apply(
        variables,
        coord=x_t[:, :, :3],
        feat=x_t[:, :, 3:] if x_t.shape[2] > 3 else None,
        t=t,
        mutable=False,
        cond=jnp.zeros_like(cond),
        point_cond_mask=point_cond_mask,
    )
    pred = pred + guidance_scale[:, None, None] * (pred - pred_no_cond)
  return pred


def generate_step_midpoint(
    model: nn.Module,
    state: TrainState,
    x_t: jax.Array,
    t_start: jax.Array,
    t_end: jax.Array,
    cond: jax.Array | None = None,
    point_cond_mask: jax.Array | None = None,
    guidance_scale: jax.Array | None = None,
    guide: bool = False,
) -> jax.Array:  # TODO(riegerfr): options other than midpoint
  """Performs a single generation step using the midpoint method.

  Args:
    model: Model to use.
    state: Training state.
    x_t: Input data of shape (batch_size, n_points, 3).
    t_start: Start time of shape (1,).
    t_end: End time of shape (1,).
    cond: Conditioning for model input.
    point_cond_mask: Mask for points to condition on.
    guidance_scale: Guidance scale for guidance.
    guide: Whether to guide the model.

  Returns:
    x_t: new data
  """
  assert len(x_t.shape) == 3

  t_start = jnp.repeat(t_start, x_t.shape[0], axis=0)
  t_end = jnp.repeat(t_end, x_t.shape[0], axis=0)

  variables = {'params': state.ema_params}
  if state.batch_stats:
    variables['batch_stats'] = state.batch_stats

  pred_start = guided_apply(
      model,
      variables,
      x_t,
      t_start,
      cond=cond,
      point_cond_mask=point_cond_mask,
      guidance_scale=guidance_scale,
      guide=guide,
  )
  dt = t_end - t_start
  dt_exp = jnp.expand_dims(dt, range(1, len(x_t.shape)))
  # Using explicit midpoint ODE solver.
  x_mid = x_t + pred_start * dt_exp / 2  # pytype: disable=unsupported-operands  # jax-operator-types
  if point_cond_mask is not None:
    x_mid = (
        x_mid * (~point_cond_mask[:, :, None])
        + x_t * point_cond_mask[:, :, None]
    )
  pred_mid = guided_apply(
      model,
      variables,
      x_mid,
      t_start + dt / 2,
      cond=cond,
      point_cond_mask=point_cond_mask,
      guidance_scale=guidance_scale,
      guide=guide,
  )
  pred_end = x_t + dt_exp * pred_mid  # pytype: disable=unsupported-operands  # jax-operator-types
  if point_cond_mask is not None:
    pred_end = (
        pred_end * (~point_cond_mask[:, :, None])
        + x_t * point_cond_mask[:, :, None]
    )
  return pred_end


def generate_step_rk4(
    model: nn.Module,
    state: TrainState,
    x_t: jax.Array,
    t_start: jax.Array,
    t_end: jax.Array,
    cond: jax.Array | None = None,
    point_cond_mask: jax.Array | None = None,
    guidance_scale: jax.Array | None = None,
    guide: bool = False,
) -> jax.Array:
  """Performs a single generation step using the RK4 method."""
  assert len(x_t.shape) == 3

  t_start = jnp.repeat(t_start, x_t.shape[0], axis=0)
  t_end = jnp.repeat(t_end, x_t.shape[0], axis=0)

  variables = {'params': state.ema_params}
  if state.batch_stats:
    variables['batch_stats'] = state.batch_stats

  dt = t_end - t_start
  dt_exp = jnp.expand_dims(dt, range(1, len(x_t.shape)))

  def f(x, t):
    return guided_apply(
        model,
        variables,
        x,
        t,
        cond=cond,
        point_cond_mask=point_cond_mask,
        guidance_scale=guidance_scale,
        guide=guide,
    )

  k1 = f(x_t, t_start)

  x_t_k2 = x_t + k1 * dt_exp / 2  # pytype: disable=unsupported-operands  # jax-operator-types
  if point_cond_mask is not None:
    x_t_k2 = (
        x_t_k2 * (~point_cond_mask[:, :, None])
        + x_t * point_cond_mask[:, :, None]
    )
  k2 = f(x_t_k2, t_start + dt / 2)

  x_t_k3 = x_t + k2 * dt_exp / 2  # pytype: disable=unsupported-operands  # jax-operator-types
  if point_cond_mask is not None:
    x_t_k3 = (
        x_t_k3 * (~point_cond_mask[:, :, None])
        + x_t * point_cond_mask[:, :, None]
    )
  k3 = f(x_t_k3, t_start + dt / 2)

  x_t_k4 = x_t + k3 * dt_exp  # pytype: disable=unsupported-operands  # jax-operator-types
  if point_cond_mask is not None:
    x_t_k4 = (
        x_t_k4 * (~point_cond_mask[:, :, None])
        + x_t * point_cond_mask[:, :, None]
    )
  k4 = f(x_t_k4, t_end)

  pred_end = x_t + (k1 + 2 * k2 + 2 * k3 + k4) * dt_exp / 6  # pytype: disable=unsupported-operands  # jax-operator-types
  if point_cond_mask is not None:
    pred_end = (
        pred_end * (~point_cond_mask[:, :, None])
        + x_t * point_cond_mask[:, :, None]
    )
  return pred_end


def generate_samples(
    model: nn.Module,
    state: TrainState,
    n_samples: int,
    sample_shape: tuple[int, ...],
    rng: jax.Array | None,
    n_steps: int,
    schedule: str = 'linear',
    return_history: bool = False,
    cond: jax.Array | None = None,
    noise: jax.Array | None = None,
    point_cond_mask: jax.Array | None = None,
    guidance_scale: jax.Array | None = None,
    guide: bool = False,
    solver: str = 'midpoint',
) -> jax.Array:
  """Generates samples from the model.

  Args:
    model: Model to use.
    state: Training state.
    n_samples: Number of samples to generate.
    sample_shape: Shape of each sample.
    rng: PRNG key.
    n_steps: Number of generation steps.
    schedule: Time schedule name.
    return_history: Whether to return generation history.
    cond: Conditioning for model input.
    noise: Noise for model input (with target points at noise[point_cond_mask]).
    point_cond_mask: Mask for points to condition on.
    guidance_scale: Guidance scale for guidance.
    guide: Whether to guide the model.
    solver: The ODE solver to use ('midpoint' or 'rk4').

  Returns:
    x_gen: Generated samples (or history).
  """
  # TODO(riegerfr): guidance weight also for point_cond_mask
  # (train with masked out points?)
  assert rng is not None or noise is not None
  assert guide == (guidance_scale is not None)

  if solver == 'midpoint':
    step_fn = generate_step_midpoint
    effective_n_steps = n_steps
  elif solver == 'rk4':
    step_fn = generate_step_rk4
    effective_n_steps = n_steps // 2
  else:
    raise ValueError(f'Unknown solver: {solver}')

  if noise is None:
    x_0 = jax.random.normal(
        rng, (cond.shape[0] if cond is not None else n_samples, *sample_shape)
    )
  else:
    x_0 = noise
  timesteps = schedules.t_schedule(
      jnp.linspace(0.0, 1.0, effective_n_steps + 1), schedule
  )

  def body_fun(i, x_t):
    return step_fn(
        model,
        state,
        x_t,
        t_start=timesteps[i],
        t_end=timesteps[i + 1],
        cond=cond,
        point_cond_mask=point_cond_mask,
        guidance_scale=guidance_scale,
        guide=guide,
    )

  if return_history:
    history = [x_0]
    for i in tqdm.tqdm(range(effective_n_steps)):
      x_t = body_fun(i, history[-1])
      history.append(x_t)
    return jnp.stack(history, axis=1)
  else:
    return jax.lax.fori_loop(0, effective_n_steps, body_fun, x_0)


def log_bins(
    log_dict: dict[str, Any], aux: dict[str, Any], bins: int = 10
) -> None:
  """Logs the loss to time bins.

  Args:
    log_dict: Dictionary to store logs.
    aux: Auxiliary outputs from loss computation.
    bins: Number of time bins.
  """
  t = jax.device_get(aux['t'])
  elementwise_loss = jax.device_get(aux['elementwise_loss'])
  bin_ranges = np.linspace(0, 1, bins)
  bin_indices = np.digitize(t, bin_ranges)

  for i in range(t.shape[0]):
    log_dict['sum'][bin_indices[i]] += elementwise_loss[i]
    log_dict['count'][bin_indices[i]] += 1


def log_dict_to_scalars(
    log_dict: dict[str, Any], tag: str = 'train_loss'
) -> dict[str, Any]:
  """Converts binned logs to scalars for logging.

  Args:
    log_dict: Dictionary of binned logs.
    tag: Tag for log entries.

  Returns:
    Dictionary of scalar logs.
  """
  return {
      f'{tag}_{k:02d}': (
          (log_dict['sum'][k] / log_dict['count'][k])
          if log_dict['count'][k]
          else -1
      )
      for k in log_dict['sum']
  }


def get_optimizer(
    config: ml_collections.ConfigDict,
) -> Callable[
    ..., optax.GradientTransformation | optax.GradientTransformationExtraArgs
]:
  """Returns the optimizer based on config.

  # TODO(riegerfr): replace with
  https://source.corp.google.com/piper///depot/google3/third_party/py/connectomics/jax/training.py;l=38

  Args:
    config: Configuration dictionary.

  Returns:
    Optimizer function.
  """
  optimizers = {
      'adamw': optax.adamw,
      'prodigy': optax.contrib.prodigy,
      'schedule_free_adamw': optax.contrib.schedule_free_adamw,
  }
  return optimizers[config.optimizer]


def moment_embedding(pc: jax.Array, n: int = 4) -> jax.Array:
  """Computes point cloud embeddings from its moments.

  Args:
    pc: Point cloud data. (shape: (batch_size, n_points, 3))
    n: Number of moments to use.

  Returns:
    Point cloud embedding.
  """
  embs = []
  mean_pc = jnp.mean(pc, axis=1)
  embs.append(mean_pc)
  centered_pc = pc - mean_pc[:, None, :]
  covs = jnp.einsum('ikj,ikl->ijl', centered_pc, centered_pc) / (
      pc.shape[1] - 1
  )
  embs.append(covs[:, *jnp.tril_indices(3)].reshape(pc.shape[0], -1))

  if n >= 3:  # (hyper)skewness/kurtosis etc. along the coordinate axes
    # TODO(riegerfr): add combination of axes like for covariance?
    std_pc = jnp.std(pc, axis=1)
    for moment in range(3, n + 1):
      embs.append(
          jnp.mean(centered_pc**moment, axis=1) / (std_pc**moment + 1e-8)
      )
  return jnp.concatenate(embs, axis=1)


@functools.partial(
    jax.jit,
    static_argnames=(
        'coord_scale',
        'feat_scale',
        'n_points',
        'cond_mode',
        'reorder_pc_type',
        'clip',
        'dst_index_cond',
        'n_combine_samples',
        'use_feat',
        'add_dummy_cond',
    ),
)
def prep_data(
    batch: dict[str, jax.Array],
    coord_scale: float,
    feat_scale: float,
    n_points: int,
    cond_mode: str = 'no',
    reorder_pc_type: str = 'no',
    clip: bool = False,
    dst_index_cond: bool = False,
    n_combine_samples: int = 1,
    use_feat: bool = False,
    add_dummy_cond: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
  """Preprocesses the data.

  Scales the coordinates by `data_scale` and subsamples to `n_points`.

  Args:
    batch: Batch of data.
    coord_scale: Scaling factor for coordinates.
    feat_scale: Scaling factor for features.
    n_points: Number of points to subsample.
    cond_mode: Conditioning mode.
    reorder_pc_type: How to reorder the point cloud.
    clip: Whether to clip the coordinates to [-1, 1].
    dst_index_cond: Whether to condition on the dataset index.
    n_combine_samples: Number of samples to combine.
    use_feat: Whether to use features.
    add_dummy_cond: Whether to add a dummy feature to cond.

  Returns:
    Preprocessed data.
  """
  coord = batch['coord'] * coord_scale
  feat = batch['feat'] * feat_scale if batch['feat'] is not None else None

  if clip:
    coord = jnp.clip(coord, -1.0, 1.0)
  coord, idx = spatial.subsample_points(
      coord, n_points
  )  # TODO(riegerfr): use rng here for additional augmentation
  feat = (
      point.batch_lookup(feat, idx[..., None])[..., 0, :] if use_feat else None
  )

  coord = reorder.reorder_named(coord, reorder_pc_type)

  if n_combine_samples > 1:
    coord = coord.reshape(
        coord.shape[0] // n_combine_samples,
        coord.shape[1] * n_combine_samples,
        coord.shape[2],
    )
    if feat is not None:
      feat = feat.reshape(
          feat.shape[0] // n_combine_samples,
          feat.shape[1] * n_combine_samples,
          feat.shape[2],
      )

  if cond_mode == 'no':
    cond = None
  else:
    if reorder_pc_type != 'no':
      raise ValueError(
          f'Reordering not supported for conditioning: {reorder_pc_type}'
      )
    if 'mean_cov' in cond_mode:
      cond = moment_embedding(coord, n=2)
    elif 'moment' in cond_mode:
      cond = moment_embedding(coord, n=4)
    elif cond_mode.startswith('fps'):
      n_points_cond = int(cond_mode.split('_')[1])
      cond = reorder_pc_x(coord[:, :n_points_cond]).reshape(coord.shape[0], -1)
    elif cond_mode.startswith('rand'):
      n_points_cond = int(cond_mode.split('_')[1])
      coord_shuffled = jax.random.permutation(
          jax.random.PRNGKey(0),  # TODO(riegerfr): use rng here
          coord,
          axis=1,
          independent=True,
      )
      cond = reorder_pc_x(coord_shuffled[:, :n_points_cond]).reshape(
          coord.shape[0], -1
      )
    else:
      raise ValueError(f'Unknown conditioning mode: {cond_mode}')

    if 'mst_leaves' in cond_mode:
      # Subsample: 256 is a tradeoff between computation time and MST
      # meaningfulness.
      coords_sub = coord[:, :256]
      dists = jnp.sum(
          (coords_sub[:, None, :] - coords_sub[:, :, None]) ** 2,
          axis=-1,
      )
      mst_adj = prim_mst(dists)
      n_leaves = (mst_adj.sum(axis=-1) == 1).sum(axis=-1)
      cond = jnp.concatenate((cond, n_leaves[:, None]), axis=1)

    if dst_index_cond:
      if n_combine_samples <= 1:  # TODO(riegerfr): Support this for all.

        cond = jnp.concatenate(
            (cond, batch['_dataset_index'][:, None] * 2 - 1), axis=1
        )
    elif add_dummy_cond:
      cond = jnp.concatenate(
          (cond, jnp.zeros((cond.shape[0], 1), dtype=cond.dtype)), axis=1
      )

  return coord, feat, cond  # TODO(riegerfr): handle feat reordering


def plot_point_clouds(
    data: jax.Array,
    size: float = 1.0,
    num_black_points: int = -1,
    plot_mst: bool = False,
    n_combine_samples: int = 1,
    color_map_name: str = 'plotly',
    color_map_multiplier: int = 37,
) -> go.Figure:
  """Plots point clouds.

  Args:
    data: Point cloud data.
    size: Size of each point.
    num_black_points: Number of points to color black. If negative, all points
      are colored according to the color map.
    plot_mst: Whether to compute and plot the Minimum Spanning Tree (MST) for
      each point cloud.
    n_combine_samples: Number of samples to combine. If greater than 1, the
      points are colored according to the mask.
    color_map_name: Name of the color map to use. 'plotly' for default plotly
      colors, or a matplotlib colormap name like 'tab20'.
    color_map_multiplier: Multiplier for color map index.

  Returns:
    Plotly figure.
  """
  data = np.asarray(data)
  if data.ndim == 2:
    data = data[None, ...]

  n_point_clouds = data.shape[0]
  n_points = data.shape[1]

  if color_map_name == 'plotly':
    color_map = px.colors.qualitative.Plotly
    use_multiplier = False
  else:
    cmap = plt.get_cmap(color_map_name)
    color_map = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]
    use_multiplier = True
  scatter_plots = []
  for i in range(n_point_clouds):
    points = data[i]
    if n_combine_samples > 1:
      combine_map = mogen_utils.get_combine_map(n_combine_samples, n_points)
      unique_mask_values = np.unique(combine_map[i])
      for mask_value in unique_mask_values:
        mask_indices = np.where(combine_map[i] == mask_value)[0]
        color_idx = (
            mask_value * color_map_multiplier if use_multiplier else mask_value
        )
        marker_color = color_map[color_idx % len(color_map)]
        if num_black_points > 0:
          marker_color_list = [marker_color] * len(mask_indices)
          for j in range(min(num_black_points, len(mask_indices))):
            marker_color_list[j] = '#000000'
        else:
          marker_color_list = marker_color
        scatter_plots.append(
            go.Scatter3d(
                x=points[mask_indices, 0],
                y=points[mask_indices, 1],
                z=points[mask_indices, 2],
                mode='markers',
                marker=dict(size=size, color=marker_color_list),
                name=f'PC {i} Mask {mask_value}',
            )
        )
    else:
      color_idx = i * color_map_multiplier if use_multiplier else i
      default_color = color_map[color_idx % len(color_map)]
      if num_black_points > 0:
        marker_color = [default_color] * n_points
        marker_color[:num_black_points] = ['#000000'] * num_black_points
      else:
        marker_color = default_color

      scatter_plots.append(
          go.Scatter3d(
              x=points[:, 0],
              y=points[:, 1],
              z=points[:, 2],
              mode='markers',
              marker=dict(size=size, color=marker_color),
              name=f'Example {i}',
          )
      )

    if plot_mst:
      dists = jnp.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
      mst_adj = np.asarray(prim_mst(dists[None, :, :])[0])  # Remove batch dim
      color_idx = i * color_map_multiplier if use_multiplier else i
      default_color = color_map[color_idx % len(color_map)]
      for j in range(n_points):
        for k in range(j + 1, n_points):
          if mst_adj[j, k] > 0:
            line_x = [points[j, 0], points[k, 0]]
            line_y = [points[j, 1], points[k, 1]]
            line_z = [points[j, 2], points[k, 2]]
            scatter_plots.append(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode='lines',
                    line=dict(color=default_color, width=2),
                    name=f'PC {i} MST Edge',
                    showlegend=(j == 0 and k == 1),
                )
            )

  fig = go.Figure(data=scatter_plots)
  fig.update_layout(
      width=750,
      height=750,
      scene=dict(
          xaxis=dict(range=[-1, 1]),
          yaxis=dict(range=[-1, 1]),
          zaxis=dict(range=[-1, 1]),
          aspectmode='cube',
      ),
      showlegend=True,
  )

  return fig


def reorder_pc_x(pc: jax.Array) -> jax.Array:
  """Reorders point clouds by the first coordinate."""
  return pc[jnp.argsort(pc[:, 0])]


def generate_and_plot_point_clouds(
    p_generate: Callable[..., jax.Array],
    state: TrainState,
    rng: jax.Array | None,
    cond: jax.Array | None,
    noise: jax.Array,
    point_cond_mask: jax.Array | None,
    workdir: epath.Path,
    filename_suffix: str,
    guidance_scale: jax.Array | None = None,
    num_black_points: int = -1,
    n_combine_samples: int = 1,
    num_devices: int = 1,
) -> None:
  """Generates samples and plots them as point clouds.

  Args:
    p_generate: JITed generate function.
    state: Training state.
    rng: PRNG key.
    cond: Conditioning.
    noise: Noise.
    point_cond_mask: Point conditioning mask.
    workdir: Working directory.
    filename_suffix: Filename suffix for output file.
    guidance_scale: Guidance scale.
    num_black_points: Number of points to color black. If negative, all points
      are colored according to the color map.
    n_combine_samples: Number of samples to combine.
    num_devices: Number of devices.
  """
  n_devices = num_devices
  bs = noise.shape[0]
  needs_repeat = bs < n_devices and n_devices % bs == 0
  if needs_repeat:
    repeats = n_devices // bs
    if cond is not None:
      cond = jnp.repeat(cond, repeats, axis=0)
    if noise is not None:
      noise = jnp.repeat(noise, repeats, axis=0)
    if point_cond_mask is not None:
      point_cond_mask = jnp.repeat(point_cond_mask, repeats, axis=0)
    if guidance_scale is not None:
      guidance_scale = jnp.repeat(guidance_scale, repeats, axis=0)

  x_gen = p_generate(
      state,
      rng,
      cond,
      noise,
      point_cond_mask,
      guidance_scale,
      guidance_scale is not None,
  )
  if needs_repeat:
    x_gen = x_gen[:bs]
  if jax.process_index() == 0:
    with storage.atomic_file(str(workdir / filename_suffix), 'w') as f:
      plot_point_clouds(
          x_gen,
          num_black_points=num_black_points,
          n_combine_samples=n_combine_samples,
      ).write_html(f)


def log_point_cond(
    state: TrainState,
    cond: jax.Array,
    init_coord: jax.Array,
    init_feat: jax.Array | None,
    workdir: epath.Path,
    config: ml_collections.ConfigDict,
    generation_rng: jax.Array,
    p_generate: Callable[..., jax.Array],
) -> None:
  """Logs point conditioning samples.

  Args:
    state: Training state.
    cond: Conditioning.
    init_coord: Initial coordinates.
    init_feat: Initial features.
    workdir: Working directory.
    config: Configuration.
    generation_rng: PRNG key for generation.
    p_generate: JITed generate function.
  """
  n_samples = min(N_PLOT, init_coord.shape[0])
  point_cond_mask = jnp.zeros((n_samples, init_coord.shape[1]), dtype=bool)
  point_cond_mask = point_cond_mask.at[:, : config.point_cond].set(True)

  sample_shape = init_coord.shape[1:]
  if init_feat is not None:
    sample_shape = (
        init_coord.shape[1],
        init_coord.shape[2] + init_feat.shape[2],
    )

  noise_all_different = jax.random.normal(
      jax.random.fold_in(generation_rng, 1337),
      (n_samples, *sample_shape),
  )
  noise_all_same = jnp.repeat(noise_all_different[:1], n_samples, axis=0)

  x_1_for_cond = init_coord[:n_samples]
  if init_feat is not None:
    x_1_for_cond = jnp.concatenate(
        (x_1_for_cond, init_feat[:n_samples]), axis=-1
    )

  same_noise_different_cond = (
      noise_all_same * (~point_cond_mask[:, :, None])
      + x_1_for_cond * point_cond_mask[:, :, None]
  )

  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=cond[:n_samples],
      noise=same_noise_different_cond,
      point_cond_mask=point_cond_mask,
      workdir=workdir,
      filename_suffix='best_mmd_train_pc_same_noise_different_cond.html',
      p_generate=p_generate,
      num_black_points=config.point_cond,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
  )

  x_1_same_cond = jnp.repeat(init_coord[:1], n_samples, axis=0)
  if init_feat is not None:
    x_1_same_cond = jnp.concatenate(
        (x_1_same_cond, jnp.repeat(init_feat[:1], n_samples, axis=0)), axis=-1
    )

  different_noise_same_cond = (
      noise_all_different * (~point_cond_mask[:, :, None])
      + x_1_same_cond * point_cond_mask[:, :, None]
  )

  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=cond[:n_samples],
      noise=different_noise_same_cond,
      point_cond_mask=point_cond_mask,
      workdir=workdir,
      filename_suffix='best_mmd_train_pc_different_noise_same_cond.html',
      p_generate=p_generate,
      num_black_points=config.point_cond,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
  )
  pcm = np.zeros((N_PLOT, init_coord.shape[1]), dtype=bool)
  for i, n in enumerate(
      [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
  ):
    pcm[i, :n] = True
  pcm = jnp.array(pcm)
  same_noise_subset_cond = jnp.repeat(noise_all_different[:1], N_PLOT, axis=0)

  x_1_subset_cond = jnp.repeat(init_coord[:1], N_PLOT, axis=0)
  if init_feat is not None:
    x_1_subset_cond = jnp.concatenate(
        (x_1_subset_cond, jnp.repeat(init_feat[:1], N_PLOT, axis=0)), axis=-1
    )

  same_noise_subset_cond = same_noise_subset_cond * (~pcm[:, :, None]) + (
      x_1_subset_cond * pcm[:, :, None]
  )

  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=jnp.zeros((N_PLOT, cond.shape[1])) if cond is not None else None,
      noise=same_noise_subset_cond,
      point_cond_mask=pcm,
      workdir=workdir,
      filename_suffix='best_mmd_train_pc_subset_cond.html',
      p_generate=p_generate,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
      # TODO(riegerfr): color first
  )

  interpolated_cond_a = init_coord[
      0, :8
  ]  # use small subset of points for better interpolation.
  interpolated_cond_b = init_coord[1, :8]

  interpolated_cond_a = reorder_pc_x(interpolated_cond_a)
  interpolated_cond_b = reorder_pc_x(interpolated_cond_b)
  interpolation_weight = jnp.arange(n_samples) / (n_samples - 1)
  interpolated_coord = (
      interpolation_weight[:, None, None] * interpolated_cond_a[None]
      + (1 - interpolation_weight)[:, None, None] * interpolated_cond_b[None]
  )

  noise_interp_cond = noise_all_same.at[:, :8, :3].set(interpolated_coord)

  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=jnp.zeros((point_cond_mask.shape[0], cond.shape[1]))
      if cond is not None
      else None,
      noise=noise_interp_cond,
      point_cond_mask=jnp.zeros_like(point_cond_mask).at[:, :8].set(True),
      workdir=workdir,
      filename_suffix='best_mmd_train_pc_interp_cond.html',
      p_generate=p_generate,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
  )


def log_conditioned(
    generation_rng: jax.Array,
    state: TrainState,
    init_coord: jax.Array,
    init_feat: jax.Array | None,
    init_cond: jax.Array,
    workdir: epath.Path,
    config: ml_collections.ConfigDict,
    p_generate: Callable[..., jax.Array],
) -> None:
  """Logs conditioned samples.

  Args:
    generation_rng: PRNG key for generation.
    state: Training state.
    init_coord: Initial coordinates.
    init_feat: Initial features.
    init_cond: Initial conditioning.
    workdir: Working directory.
    config: Configuration.
    p_generate: JITed generate function.
  """
  sample_shape = init_coord.shape[1:]
  if init_feat is not None:
    sample_shape = (
        init_coord.shape[1],
        init_coord.shape[2] + init_feat.shape[2],
    )

  noise_all_different = jax.random.normal(
      jax.random.fold_in(generation_rng, 1337),
      (N_PLOT, *sample_shape),
  )
  if config.point_cond > 0:
    point_cond_mask = jnp.zeros(
        (N_PLOT, init_coord.shape[1]),
        dtype=bool,
    )
  else:
    point_cond_mask = None

  cur_cond = jnp.repeat(init_cond[:1], noise_all_different.shape[0], axis=0)
  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=jnp.concatenate((cur_cond, jnp.ones_like(cur_cond)), axis=-1),
      noise=noise_all_different,
      point_cond_mask=point_cond_mask,
      workdir=workdir,
      filename_suffix='best_mmd_train_cond_same_noise_different.html',
      p_generate=p_generate,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
  )

  noise_all_same = jnp.repeat(noise_all_different[:1], N_PLOT, axis=0)
  n_samples = min(N_PLOT, init_coord.shape[0])
  cur_cond = init_cond[:n_samples]
  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=jnp.concatenate((cur_cond, jnp.ones_like(cur_cond)), axis=-1),
      noise=noise_all_same[:n_samples],
      point_cond_mask=point_cond_mask[:n_samples]
      if point_cond_mask is not None
      else None,
      workdir=workdir,
      filename_suffix='best_mmd_train_cond_different_noise_same.html',
      p_generate=p_generate,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
  )
  guidance_scales = jnp.array([
      -1.0,
      -0.5,
      -0.2,
      -0.1,
      0.0,
      0.1,
      0.2,
      0.5,
      1.0,
      2.0,
      5.0,
      10.0,
      20.0,
      50.0,
      100.0,
      200.0,
  ])
  # generate (all noises are the same, cond the same)
  cur_cond = jnp.repeat(init_cond[:1], N_PLOT, axis=0)
  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=jnp.concatenate((cur_cond, jnp.ones_like(cur_cond)), axis=-1),
      noise=jnp.repeat(noise_all_same[:1], N_PLOT, axis=0),
      point_cond_mask=jnp.repeat(point_cond_mask[:1], N_PLOT, axis=0)
      if point_cond_mask is not None
      else None,
      workdir=workdir,
      filename_suffix='best_mmd_train_cond_same_noise_same_guidance.html',
      guidance_scale=guidance_scales,
      p_generate=p_generate,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
  )
  arange = jnp.arange(N_PLOT) / (N_PLOT - 1)
  interpolated_init_cond = (
      arange[:, None] * init_cond[0] + (1 - arange)[:, None] * init_cond[1]
  )

  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=jnp.concatenate(
          (interpolated_init_cond, jnp.ones_like(interpolated_init_cond)),
          axis=-1,
      ),
      noise=noise_all_same,
      point_cond_mask=point_cond_mask,
      workdir=workdir,
      filename_suffix='best_mmd_train_cond_interp_first_two_noise_same.html',
      p_generate=p_generate,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
  )

  cond_neg = jnp.zeros_like(init_cond[0])
  cond_neg = cond_neg.at[0].set(-2.0)
  cond_pos = jnp.zeros_like(init_cond[0])
  cond_pos = cond_pos.at[0].set(2.0)
  cur_cond = arange[:, None] * cond_neg + (1 - arange)[:, None] * cond_pos

  generate_and_plot_point_clouds(
      state=state,
      rng=None,
      cond=jnp.concatenate(
          (cur_cond, jnp.zeros_like(cur_cond).at[:, 0].set(1.0)), axis=-1
      ),
      noise=noise_all_same,
      point_cond_mask=point_cond_mask,
      workdir=workdir,
      filename_suffix='best_mmd_train_cond_interp_first_pc_noise_same.html',
      p_generate=p_generate,
      n_combine_samples=config.n_combine_samples,
      num_devices=config.num_devices,
  )
  # TODO(riegerfr): log (gen_x.mean(axis=1)[:,0] -
  # cur_cond[:, 0]).square().mean() as "condition_error"

  if config.train_set == 'mixed':

    cond_class0 = jnp.zeros_like(init_cond[0])
    cond_class0 = cond_class0.at[-1].set(-1.0)
    cond_class1 = jnp.zeros_like(init_cond[0])
    cond_class1 = cond_class1.at[-1].set(1.0)
    cur_cond = (
        arange[:, None] * cond_class0 + (1 - arange)[:, None] * cond_class1
    )
    generate_and_plot_point_clouds(
        state=state,
        rng=None,
        cond=jnp.concatenate((cur_cond, jnp.ones_like(cur_cond)), axis=-1),
        noise=noise_all_same,
        point_cond_mask=point_cond_mask,
        workdir=workdir,
        filename_suffix='cond_interp_last_zero_one_noise_same.html',
        p_generate=p_generate,
        n_combine_samples=config.n_combine_samples,
        num_devices=config.num_devices,
    )

    # last cond dim is class (0 or 1)
    # 1 batch with class 0, 1 batch with class 1
    all_class0_cond = jnp.zeros_like(init_cond[:N_PLOT]).at[:, -1].set(-1.0)
    all_class1_cond = jnp.zeros_like(init_cond[:N_PLOT]).at[:, -1].set(1.0)
    cur_cond = jnp.where(
        arange[:, None] < 0.5, all_class0_cond, all_class1_cond
    )
    generate_and_plot_point_clouds(
        state=state,
        rng=None,
        cond=jnp.concatenate((cur_cond, jnp.ones_like(cur_cond)), axis=-1),
        noise=noise_all_different,
        point_cond_mask=point_cond_mask,
        workdir=workdir,
        filename_suffix='cond_interp_diff_class_diff_noise.html',
        p_generate=p_generate,
        n_combine_samples=config.n_combine_samples,
        num_devices=config.num_devices,
    )

  if 'mst_leaves' in config.cond_mode:
    # TODO(riegerfr): handle better (not just for mixed, find index easily).
    cond_second_last_one = jnp.zeros_like(init_cond[0])
    cond_second_last_one = cond_second_last_one.at[-2].set(1.0)
    cur_cond = (
        arange[:, None] * cond_second_last_one
        + (1 - arange)[:, None] * jnp.zeros_like(cond_second_last_one)
    ) * 256  # 0 is no leaves, 256 is only leaves
    generate_and_plot_point_clouds(
        state=state,
        rng=None,
        cond=jnp.concatenate(
            (cur_cond, jnp.zeros_like(cur_cond).at[-2].set(1.0)), axis=-1
        ),
        noise=noise_all_same,
        point_cond_mask=point_cond_mask,
        workdir=workdir,
        filename_suffix='cond_interp_leaves_noise_same.html',
        p_generate=p_generate,
        n_combine_samples=config.n_combine_samples,
        num_devices=config.num_devices,
    )


@jax.jit
def prim_mst(adj: jax.Array) -> jax.Array:
  """Computes the minimum spanning tree of a graph.

  Following https://en.wikipedia.org/wiki/Prim%27s_algorithm

  Args:
    adj: Adjacency matrix of the graph.

  Returns:
    Binary adjacency matrix of the minimum spanning tree.
  """
  batch_size, num_nodes = adj.shape[0], adj.shape[1]
  cheapest_cost = jnp.full((batch_size, num_nodes), jnp.inf, dtype=adj.dtype)
  cheapest_edge_parent = jnp.full((batch_size, num_nodes), -1, dtype=jnp.int32)
  cheapest_cost = cheapest_cost.at[:, 0].set(0.0)
  explored_mask = jnp.zeros((batch_size, num_nodes), dtype=bool)
  initial_state = (explored_mask, cheapest_cost, cheapest_edge_parent)
  batch_indices = jnp.arange(batch_size)

  def loop_body(_, current_loop_state):
    (
        current_explored_mask,
        current_cheapest_cost,
        current_cheapest_edge_parent,
    ) = current_loop_state
    cost_for_unexplored = jnp.where(
        current_explored_mask, jnp.inf, current_cheapest_cost
    )
    current_vertex = jnp.argmin(cost_for_unexplored, axis=1)
    next_explored_mask = current_explored_mask.at[
        batch_indices, current_vertex
    ].set(True)
    weights_from_current_vertex = adj[batch_indices, current_vertex, :]
    needs_update_for_neighbor = (~next_explored_mask) & (
        weights_from_current_vertex < current_cheapest_cost
    )
    updated_cheapest_cost = jnp.where(
        needs_update_for_neighbor,
        weights_from_current_vertex,
        current_cheapest_cost,
    )
    updated_cheapest_edge_parent = jnp.where(
        needs_update_for_neighbor,
        current_vertex[:, None],
        current_cheapest_edge_parent,
    )
    return (
        next_explored_mask,
        updated_cheapest_cost,
        updated_cheapest_edge_parent,
    )

  _, _, final_cheapest_edge_parent = jax.lax.fori_loop(
      0, num_nodes, loop_body, initial_state
  )
  mst_adj = jnp.zeros_like(adj)
  total_edges_for_nonzero = batch_size * max(0, num_nodes - 1)
  b_indices_for_edges, child_nodes_with_edges = jnp.nonzero(
      (final_cheapest_edge_parent != -1),
      size=total_edges_for_nonzero,
      fill_value=0,
  )
  parent_nodes_for_edges = final_cheapest_edge_parent[
      b_indices_for_edges, child_nodes_with_edges
  ]
  mst_adj = mst_adj.at[
      b_indices_for_edges, child_nodes_with_edges, parent_nodes_for_edges
  ].set(1.0)
  mst_adj = mst_adj.at[
      b_indices_for_edges, parent_nodes_for_edges, child_nodes_with_edges
  ].set(1.0)
  return mst_adj


@functools.partial(jax.jit, static_argnames=('mst',))
def simple_embs(pc: jax.Array, mst: bool = False) -> jax.Array:
  """Computes simple embeddings for a point cloud.

  Args:
    pc: Point cloud.
    mst: Whether to use the minimum spanning tree.

  Returns:
    Simple embeddings.
  """
  mean_pc = jax.numpy.mean(pc, axis=1)
  origin_dist = (mean_pc**2).sum(-1, keepdims=True) ** 0.5
  centered_pc = pc - mean_pc[:, None, :]
  covs = jax.numpy.einsum('ikj,ikl->ijl', centered_pc, centered_pc) / (
      pc.shape[1] - 1
  )
  cov_eigval = jax.numpy.linalg.eigvalsh(covs) ** 0.5  # std along PCs
  # TODO(riegerfr): higher order moments?

  if pc.shape[1] <= 8192:
    dists = ((pc[:, :, None] - pc[:, None, :]) ** 2).sum(-1) ** 0.5
    max_dists = dists.max(axis=-1)

    min_dists = -jax.lax.top_k(-dists, k=2)[0][..., 1]  # dists.min(axis=-1)
    mean_min_dists = min_dists.mean(axis=-1, keepdims=True)
    mean_max_dists = max_dists.mean(axis=-1, keepdims=True)
    std_min_dists = min_dists.std(axis=-1, keepdims=True)
    std_max_dists = max_dists.std(axis=-1, keepdims=True)
  else:
    dists, _ = spatial.kdnn(pc, 2)
    min_dists = dists[..., 1]  # k=1 is the point itself
    mean_min_dists = min_dists.mean(axis=-1, keepdims=True)
    std_min_dists = min_dists.std(axis=-1, keepdims=True)
    #  max_dists difficult to compute for k-d tree -> subsample
    pc_subsample = spatial.subsample_points(pc, num=8192)[0]
    dists = ((pc_subsample[:, :, None] - pc_subsample[:, None, :]) ** 2).sum(
        -1
    ) ** 0.5
    max_dists = dists.max(axis=-1)
    mean_max_dists = max_dists.mean(axis=-1, keepdims=True)
    std_max_dists = max_dists.std(axis=-1, keepdims=True)
  # TODO(riegerfr): Divide by n_points-1 for mean edge length.

  feats = [
      origin_dist,
      cov_eigval,
      mean_min_dists,
      mean_max_dists,
      std_min_dists,
      std_max_dists,
  ]
  if mst:
    mst_weights = prim_mst(dists) * dists
    mst_max = mst_weights.max((-2, -1))[:, None]
    mst_sum = mst_weights.sum(axis=(-2, -1))[:, None]
    # TODO(riegerfr): also std (but need to select only mst edges, same for
    # mean (instead of sum))?
    feats.append(mst_max)
    feats.append(mst_sum)

  return jax.numpy.concatenate(
      feats,
      axis=-1,
  )


def compute_metrics(
    x_gen: jax.Array,
    cfg: ml_collections.ConfigDict,
    norm_std: bool = True,
    mst: bool = True,
    subset_size: int = 16384,
    # return_subset_metrics: bool = False,
) -> tuple[float, float, float, float, float, float]:
  """Computes metrics for generated points.

  Args:
    x_gen: Generated points.
    cfg: Configuration.
    norm_std: Whether to normalize the embeddings by the standard deviation of
      the training set.
    mst: Whether to use the MST for the embeddings.
    subset_size: Size of the subset of the training data to use for the metrics.

  Returns:
    Tuple of MMD and FID metrics for training and validation sets, and MMD
    metrics for a subset of the embeddings.
  """
  n_combine_samples = cfg.get('n_combine_samples', 1)
  emb_path_suffix = f'_cs{n_combine_samples}' if n_combine_samples > 1 else ''
  batch_size = min(4, cfg.batch_size)
  embs_gen = jax.numpy.concatenate([
      simple_embs(x_gen[i * batch_size : (i + 1) * batch_size], mst)  # pytype: disable=wrong-arg-types
      for i in range(x_gen.shape[0] // batch_size)
  ])

  assert embs_gen.shape[0] == x_gen.shape[0]
  with open(
      (
          cfg.simple_emb_path
          + f'emb_{cfg.train_set if cfg.train_set not in ("sub1", "sub2", "all") else "train"}_{cfg.n_points}{emb_path_suffix}_train{"_mst" if mst else ""}.npz'
      ),
      'rb',
  ) as f:
    embs_train = np.load(f)['arr_0'][:subset_size]  # subset to save memory.
  with open(
      (
          cfg.simple_emb_path
          + f'emb_{cfg.train_set if cfg.train_set not in ("sub1", "sub2", "all") else "train"}_{cfg.n_points}{emb_path_suffix}_val{"_mst" if mst else ""}.npz'
      ),
      'rb',
  ) as f:
    embs_val = np.load(f)['arr_0'][:16384]
  assert embs_train.shape[1] == embs_val.shape[1] == embs_gen.shape[1]
  if norm_std:
    train_std = embs_train.std(axis=0, keepdims=True)  # pytype: disable=attribute-error
    embs_gen = embs_gen / train_std
    embs_train = embs_train / train_std
    embs_val = embs_val / train_std

  logging.info('embs_gen shape: %r', embs_gen.shape)
  logging.info('embs_train shape: %r', embs_train.shape)
  logging.info('embs_val shape: %r', embs_val.shape)
  mmd_train = distance.mmd(embs_gen, embs_train)
  mmd_val = distance.mmd(embs_gen, embs_val)
  fid_train = 10_000  # TODO(riegerfr): remove
  fid_val = 10_000
  mmd_train_sub = distance.mmd(embs_gen[:, :8], embs_train[:, :8])
  mmd_val_sub = distance.mmd(embs_gen[:, :8], embs_val[:, :8])

  return mmd_train, mmd_val, fid_train, fid_val, mmd_train_sub, mmd_val_sub


def save_point_clouds_svg(
    data: np.ndarray,
    filepath: str,
    size: float = 0.1,
    alpha: float = 1.0,
    # *,
    overlap_percentage: int = 25,
    rasterized_dpi: int = 0,
    rotate: bool = False,
) -> display.HTML:
  """Saves point clouds as an SVG image grid.

  Args:
    data: Point cloud data, as a batch of shape (num_samples, num_points, 3).
    filepath: Path to save the SVG file.
    size: Size of the marker in points^2.
    alpha: Alpha for the markers.
    overlap_percentage: How much the subplots should overlap (0-100). Defaults
      to 25.
    rasterized_dpi: DPI for rasterization.
    rotate: Whether to rotate point clouds.

  Returns:
    An HTML object containing the SVG image.
  """
  data = np.asarray(data)
  if data.ndim != 3:
    raise ValueError(
        'Input data must be 3-dimensional with shape (num_samples, n_points, 3)'
    )
  if rotate:
    data_rotated = np.zeros_like(data)
    for i in range(data.shape[0]):
      pca = decomposition.PCA(n_components=3)
      pca.fit(data[i])
      rotation_matrix = np.array(
          [pca.components_[1], pca.components_[0], pca.components_[2]]
      ).T
      data_rotated[i] = data[i] @ rotation_matrix
    data = data_rotated

  num_samples = data.shape[0]
  grid_size = math.ceil(math.sqrt(num_samples))
  color_map = plt.get_cmap('tab10')

  fig, axes = plt.subplots(
      grid_size, grid_size, figsize=(8, 8), subplot_kw={'projection': '3d'}
  )

  fig.set_facecolor('white')

  spacing = -float(overlap_percentage) / 100.0
  fig.subplots_adjust(
      left=0, right=1, bottom=0, top=1, wspace=spacing, hspace=spacing
  )

  axes = axes.flatten() if num_samples > 1 else [axes]

  for i, ax in enumerate(axes):
    ax.clear()
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()

    if i < num_samples:
      ax.set_xlim([-1, 1])
      ax.set_ylim([-1, 1])
      ax.set_zlim([-1, 1])
      ax.view_init(elev=20, azim=-90)

      point_cloud = data[i, :, :]
      color = color_map(i % color_map.N)
      ax.scatter(
          point_cloud[:, 0],
          point_cloud[:, 1],
          point_cloud[:, 2],
          s=size,
          c=[color],
          alpha=alpha,
          marker='o',
          linewidths=0,
          rasterized=rasterized_dpi > 0,
      )

  for i in range(num_samples, len(axes)):
    axes[i].set_axis_off()

  with tempfile.NamedTemporaryFile(suffix='.svg') as tmp:
    fig.savefig(
        tmp.name,
        format='svg',
        facecolor='white',
        bbox_inches='tight',
        pad_inches=0,
        dpi=rasterized_dpi if rasterized_dpi else None,
    )
    shutil.copy(tmp.name, filepath)
    tmp.seek(0)
    svg_bytes = tmp.read()

  plt.close(fig)

  base64_svg = base64.b64encode(svg_bytes).decode()
  return display.HTML(f'<img src="data:image/svg+xml;base64,{base64_svg}" />')


def save_point_clouds_gif(
    data: np.ndarray,
    filepath: str,
    size: float = 0.1,
    duration: float = 0.1,
    first_last_duration: float = 0.0,
    alpha: float = 1.0,
    # *,
    overlap_percentage: int = 25,
    dpi: int = 300,
) -> display.HTML:
  """Saves a high-quality animated GIF of point clouds in an overlapping grid.

  Args:
    data: Point cloud data, as a batch of shape (batch_size, num_timesteps,
      num_points, 3).
    filepath: Path to save the GIF file.
    size: Size of the marker in points^2.
    duration: Duration of each frame in the GIF in seconds.
    first_last_duration: Duration of the first and last frames in seconds. If
      larger than `duration`, these frames will be repeated.
    alpha: Alpha for the markers.
    overlap_percentage: How much the subplots should overlap (0-100). Defaults
      to 25.
    dpi: DPI for rasterization.

  Returns:
    An HTML object containing the animated GIF.
  """
  data = np.asarray(data)
  if data.ndim != 4:
    raise ValueError(
        'Input data must be 4-dimensional with shape (batch, time, n_points, 3)'
    )

  num_samples = data.shape[0]
  num_timesteps = data.shape[1]
  grid_size = math.ceil(math.sqrt(num_samples))
  color_map = plt.get_cmap('tab10')

  fig, axes = plt.subplots(
      grid_size, grid_size, figsize=(8, 8), subplot_kw={'projection': '3d'}
  )

  fig.set_facecolor('white')

  spacing = -float(overlap_percentage) / 100.0
  fig.subplots_adjust(
      left=0, right=1, bottom=0, top=1, wspace=spacing, hspace=spacing
  )

  axes = axes.flatten() if num_samples > 1 else [axes]

  frames_forward = list(range(num_timesteps))
  frames_backward = (
      list(range(num_timesteps - 2, 0, -1)) if num_timesteps > 1 else []
  )

  if first_last_duration > duration and num_timesteps > 0:
    n_repeats = int(first_last_duration / duration)
    frames_forward = [0] * (n_repeats - 1) + frames_forward
    if num_timesteps > 1:
      frames_forward.extend([num_timesteps - 1] * (n_repeats - 1))
  frames_sequence = frames_forward + frames_backward

  def update(frame_index):
    time_step = frames_sequence[frame_index]
    for i, ax in enumerate(axes):
      ax.clear()
      ax.patch.set_alpha(0.0)
      ax.set_axis_off()

      if i < num_samples:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(elev=20, azim=-90)

        point_cloud = data[i, time_step, :, :]
        color = color_map(i % color_map.N)
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            s=size,
            c=[color],
            alpha=alpha,
            marker='o',
            linewidths=0,
        )

  for i in range(num_samples, len(axes)):
    axes[i].set_axis_off()

  anim = animation.FuncAnimation(
      fig, update, frames=len(frames_sequence), blit=False
  )

  with tempfile.NamedTemporaryFile(suffix='.gif') as tmp:
    anim.save(
        tmp.name,
        writer='imagemagick',
        fps=1.0 / duration,
        dpi=dpi,
        savefig_kwargs={
            'facecolor': 'white',
            'bbox_inches': 'tight',
            'pad_inches': 0,
        },
    )
    shutil.copy(tmp.name, filepath)
    tmp.seek(0)
    gif_bytes = tmp.read()

  plt.close(fig)

  base64_gif = base64.b64encode(gif_bytes).decode()
  return display.HTML(f'<img src="data:image/gif;base64,{base64_gif}" />')


def save_point_clouds_rotation_gif(
    data: np.ndarray,
    filepath: str,
    duration_seconds: int = 1,
    fps: int = 3,
    size: float = 0.05,
    alpha: float = 1.0,
    dpi: int = 300,
    color_map_name: str = 'tab10',
    color_map_multiplier: int = 37,
) -> display.HTML:
  """Saves an animated GIF of rotating point clouds using Matplotlib.

  Args:
    data: Point cloud data, as a batch of shape (num_samples, num_points, 3).
    filepath: Path to save the GIF file.
    duration_seconds: Duration of the GIF in seconds.
    fps: Frames per second.
    size: Size of the marker in points^2.
    alpha: Alpha for the markers.
    dpi: Dots per inch for the GIF.
    color_map_name: Name of the matplotlib colormap to use.
    color_map_multiplier: Multiplier for color map index.

  Returns:
    An HTML object containing the animated GIF.
  """
  data = np.asarray(data)
  num_samples = data.shape[0]
  grid_size = math.ceil(math.sqrt(num_samples))
  num_frames = duration_seconds * fps
  angles = np.linspace(0, 360, num_frames, endpoint=False)

  color_map = plt.get_cmap(color_map_name)

  fig, axes = plt.subplots(
      grid_size, grid_size, figsize=(8, 8), subplot_kw={'projection': '3d'}
  )

  fig.subplots_adjust(
      left=0, right=1, bottom=0, top=1, wspace=-0.5, hspace=-0.5
  )

  axes = axes.flatten() if num_samples > 1 else [axes]

  def update(frame_index):
    angle = angles[frame_index]
    for i, ax in enumerate(axes):
      ax.clear()
      ax.patch.set_alpha(0.0)
      ax.set_axis_off()

      if i < num_samples:
        pc = data[i]
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(elev=20, azim=angle)

        color = color_map((i * color_map_multiplier) % color_map.N)
        ax.scatter(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            s=size,
            c=[color],
            alpha=alpha,
            marker='o',
            linewidths=0,
        )

  for i in range(num_samples, len(axes)):
    axes[i].set_axis_off()

  anim = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)

  with tempfile.NamedTemporaryFile(suffix='.gif') as tmp:
    anim.save(
        tmp.name,
        writer='imagemagick',
        fps=fps,
        dpi=dpi,
        savefig_kwargs={
            'transparent': True,
            'bbox_inches': 'tight',
            'pad_inches': 0,
        },
    )
    shutil.copy(tmp.name, filepath)
    tmp.seek(0)
    gif_bytes = tmp.read()

  plt.close(fig)
  return display.HTML(
      f'<img src="data:image/gif;base64,{base64.b64encode(gif_bytes).decode()}"'
      ' />'
  )

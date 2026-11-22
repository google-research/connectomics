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
"""Schedules for flow matching."""

import jax
import jax.numpy as jnp
from jax.scipy import special as jsp_special


def cosine(t: jax.Array, exponent: float = 1.0, skew: float = 1.0) -> jax.Array:
  """Cosine schedule with optional exponent.

  Args:
    t: Timestep in the range [0, 1].
    exponent: Exponent to apply to the cosine function. Default is 1.0 (standard
      cosine schedule).
    skew: Skew to apply to the cosine function. Default is 1.0.

  Returns:
    Transformed timestep in the range [0, 1].
  """
  t = t**skew
  return ((jnp.abs((jnp.abs(jnp.cos(t * jnp.pi)) - 1)) ** exponent) - 1) * (
      (t < 0.5) - 0.5
  ) + 0.5


def linear_logsnr(
    t: jax.Array, min_log_snr: float = -20.0, max_log_snr: float = 20.0
) -> jax.Array:
  """Sigmoid schedule with linear logSNR mapping.

  Args:
    t: Timestep in the range [0, 1].
    min_log_snr: Minimum logSNR value. Default is -20.0.
    max_log_snr: Maximum logSNR value. Default is 20.0.

  Returns:
    Transformed timestep in the range [0, 1].
  """
  x = t * (max_log_snr - min_log_snr) + min_log_snr
  return jax.nn.sigmoid(x)


def logit_normal(t: jax.Array, m: float = 0.0, s: float = 1.0) -> jax.Array:
  """Maps uniform t in [0, 1] to a logit-normal(m, s) variable.

  Used as a training time distribution (Esser et al. 2024, arXiv:2403.03206):
  for u ~ U[0, 1], sigmoid(m + s * Phi^-1(u)) ~ logit-normal(m, s), which puts
  more mass on intermediate times than on the endpoints. The map is monotone
  with fixed endpoints (0 -> 0, 1 -> 1).

  Args:
    t: Timestep in the range [0, 1].
    m: Location of the underlying normal (positive shifts mass to t -> 1).
    s: Scale of the underlying normal (> 0).

  Returns:
    Transformed timestep in the range [0, 1].
  """
  return jax.nn.sigmoid(m + s * jsp_special.ndtri(t))


def anneal_factor(
    step: jax.Array,
    begin: int,
    steps: int,
    final: float = 0.0,
    shape: str = 'linear',
) -> jax.Array:
  """Multiplicative annealing factor: 1 before `begin`, `final` after.

  Args:
    step: Current (train) step.
    begin: Step at which the annealing starts.
    steps: Number of annealing steps (> 0).
    final: Factor reached at `begin + steps` and kept afterwards.
    shape: 'linear' or 'cosine' interpolation between 1 and `final`.

  Returns:
    Scalar factor in [min(final, 1), max(final, 1)].

  Raises:
    ValueError: If `steps` <= 0 or `shape` is unknown.
  """
  if steps <= 0:
    raise ValueError(f'steps must be > 0, got {steps}')
  frac = jnp.clip(
      (jnp.asarray(step, jnp.float32) - float(begin)) / float(steps), 0.0, 1.0
  )
  if shape == 'linear':
    w = frac
  elif shape == 'cosine':
    w = 0.5 * (1.0 - jnp.cos(jnp.pi * frac))
  else:
    raise ValueError(f'Unknown anneal shape: {shape}')
  return 1.0 + (final - 1.0) * w


def t_schedule(t: jax.Array, s_name: str = 'linear') -> jax.Array:
  """Transforms the timestep t using the specified schedule.

  Args:
    t: Timestep in the range [0, 1].
    s_name: Name of the schedule to use. Options are: - 'linear': No
      transformation (returns t). - 'cosine': Standard cosine schedule. -
      'cosine_[exponent]': Cosine schedule with a custom exponent (e.g.,
      'cosine_2.0'). - 'linear_logsnr': Sigmoid schedule with linear logSNR
      mapping. - 'linear_logsnr_[min]_[max]': Sigmoid schedule with custom min
      and max logSNR values (e.g., 'linear_logsnr_-10.0_10.0'). -
      'logitnormal_[m]_[s]': Logit-normal(m, s) map, see `logit_normal` (e.g.,
      'logitnormal_0.0_1.0').

  Returns:
    Transformed timestep in the range [0, 1].

  Raises:
    ValueError: If the specified schedule name is unknown.
  """
  if s_name in ('linear', 'uniform'):
    return t
  elif s_name.startswith('cosine'):
    if s_name == 'cosine':
      return cosine(t)
    elif len(s_name.split('_')) == 2:
      exponent = float(s_name.split('_')[1])
      return cosine(t, exponent)
    else:
      exponent, skew = map(float, s_name.split('_')[1:])
      return cosine(t, exponent, skew)
  elif s_name.startswith('linear_logsnr'):
    if s_name == 'linear_logsnr':
      return linear_logsnr(t)
    else:
      min_log_snr, max_log_snr = map(float, s_name.split('_')[-2:])
      return linear_logsnr(t, min_log_snr, max_log_snr)
  elif s_name.startswith('logitnormal_'):
    m, s = map(float, s_name.split('_')[1:])
    return logit_normal(t, m, s)
  elif s_name.startswith('rho'):
    rho_str = s_name[3:].lstrip('_')
    rho = float(rho_str) if rho_str else 7.0
    return 1.0 - (1.0 - t) ** rho
  elif s_name.startswith('edm_'):
    rho = float(s_name.split('_')[1])
    return 1.0 - (1.0 - t) ** rho
  else:
    raise ValueError(f'Unknown schedule: {s_name}')

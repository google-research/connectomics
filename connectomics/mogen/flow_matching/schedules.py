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


def t_schedule(t: jax.Array, s_name: str = 'linear') -> jax.Array:
  """Transforms the timestep t using the specified schedule.

  Args:
    t: Timestep in the range [0, 1].
    s_name: Name of the schedule to use. Options are:
      - 'linear': No transformation (returns t).
      - 'cosine': Standard cosine schedule.
      - 'cosine_[exponent]': Cosine schedule with a custom exponent (e.g.,
        'cosine_2.0').
      - 'linear_logsnr': Sigmoid schedule with linear logSNR mapping.
      - 'linear_logsnr_[min]_[max]': Sigmoid schedule with custom min and max
        logSNR values (e.g., 'linear_logsnr_-10.0_10.0').

  Returns:
    Transformed timestep in the range [0, 1].

  Raises:
    ValueError: If the specified schedule name is unknown.
  """
  if s_name == 'linear':
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
  else:
    raise ValueError(f'Unknown schedule: {s_name}')

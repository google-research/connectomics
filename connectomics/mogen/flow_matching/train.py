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
"""Flow Matching training."""

import collections
from collections.abc import Mapping
import json
import os
import re
import time
from typing import Any, Sequence

from absl import flags
from absl import logging

try:
  flags.DEFINE_string(
      'ckpt_workdir',
      None,
      'Directory to store heavy Orbax checkpoints locally in the compute cell.',
  )
except flags.DuplicateFlagError:
  pass
try:
  flags.DEFINE_string(
      'ckpt_dir_override',
      None,
      'Direct directory to store checkpoints.',
  )
except flags.DuplicateFlagError:
  pass
try:
  flags.DEFINE_string(
      'ckpt_root_dir',
      None,
      'Root directory (e.g. /cns/nf-d/home/riegerfr/point/fm_ckpts) to store'
      ' checkpoints.',
  )
except flags.DuplicateFlagError:
  pass
try:
  flags.DEFINE_string(
      'model_soup_ckpts',
      '',
      'Comma- or "+"-separated list of checkpoint directories for model soup'
      ' averaging.',
  )
  flags.DEFINE_string(
      'model_soup_steps',
      '',
      'Comma- or "+"-separated list of steps corresponding to'
      ' model_soup_ckpts.',
  )
  flags.DEFINE_string(
      'model_soup_weights',
      '',
      'Comma- or "+"-separated list of weights summing to 1.0 for model soup'
      ' averaging.',
  )
  flags.DEFINE_integer(
      'save_checkpoint_secs',
      0,
      'Wall-clock interval in seconds to trigger a checkpoint save (0 to'
      ' disable).',
  )
except flags.DuplicateFlagError:
  pass

from clu import metric_writers
from connectomics.mogen.flow_matching import schedules
from connectomics.mogen.flow_matching import utils
from e3x.so3 import rotations
from etils import epath
from ffn.inference import storage
import grain.tensorflow as grain
import jax
from jax import sharding
from jax.experimental import multihost_utils
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import tqdm

def _random_batch(batch_size, n_points, n_feat, rng):
  """Generates a random batch of point cloud data."""
  coord = rng.standard_normal((batch_size, n_points, 3)).astype(np.float32)
  coord = coord / np.abs(coord).max() * 0.9  # Scale to [-0.9, 0.9]
  feat = rng.standard_normal((batch_size, n_points, n_feat)).astype(
      np.float32
  )
  return {'coord': coord, 'feat': feat}


def _get_fake_dataloaders(
    per_device_batch_size,
    n_points=2048,
    n_feat=3,
    n_train_batches=10,
    n_val_batches=2,
    seed=0,
    num_devices=1,
):
  """Returns fake dataloaders generating random point cloud data."""
  batch_size = per_device_batch_size * num_devices
  rng = np.random.RandomState(seed)
  train_data = [
      _random_batch(batch_size, n_points, n_feat, rng)
      for _ in range(n_train_batches)
  ]
  val_data = [
      _random_batch(batch_size, n_points, n_feat, rng)
      for _ in range(n_val_batches)
  ]

  def _make_loader(batches):
    def _iter_fn():
      while True:
        yield from batches

    return _iter_fn()

  return (
      _make_loader(train_data),
      len(train_data) * batch_size,
      _make_loader(val_data),
      len(val_data) * batch_size,
      None,
  )


def should_save_checkpoint(
    step: int,
    save_checkpoint_steps: int,
    last_checkpoint_time: float,
    save_checkpoint_secs: int,
    now: float,
    max_steps: int,
    warm_sanity: bool = False,
) -> tuple[bool, bool]:
  """Determines if a checkpoint should be saved.

  Args:
    step: Current training step.
    save_checkpoint_steps: Step interval to save checkpoints (0 to disable).
    last_checkpoint_time: Timestamp of last saved checkpoint.
    save_checkpoint_secs: Wall-clock interval in seconds (0 to disable).
    now: Current timestamp.
    max_steps: Maximum training steps.
    warm_sanity: Whether warm-start sanity checkpoint is needed.

  Returns:
    (should_save, is_wall_clock)
  """
  time_to_save = bool(
      save_checkpoint_secs > 0
      and (now - last_checkpoint_time >= save_checkpoint_secs)
  )
  step_to_save = bool(
      (save_checkpoint_steps > 0 and step % save_checkpoint_steps == 0)
      or step == max_steps
      or step == 10
      or step == 20
      or warm_sanity
  )
  return (step_to_save or time_to_save), time_to_save


_REMAT_PREFIX = 'Checkpoint'


def _strip_remat_prefix(key: Any) -> Any:
  """Drops the `nn.remat` name prefix: 'CheckpointFoo_0' -> 'Foo_0'."""
  if not isinstance(key, str):
    return key
  while key.startswith(_REMAT_PREFIX) and len(key) > len(_REMAT_PREFIX):
    key = key[len(_REMAT_PREFIX) :]
  return key


def _align_remat_names(template: Any, restored: Any) -> Any:
  """Renames `restored` dict keys to match `template` modulo remat prefixes.

  Flax's `nn.remat(Module)` names submodules 'Checkpoint<Module>_<i>', so a
  checkpoint saved with `use_remat=True` has e.g.
  'CheckpointPointInfinityBlock_0' where a model built with `use_remat=False`
  expects 'PointInfinityBlock_0' (and vice versa). The parameters themselves
  are identical. Keys that match exactly are kept; otherwise a unique match
  after stripping the prefix is used. Unmatched restored keys are kept as-is so
  any real structure mismatch still surfaces in the caller's `tree_map`.

  Args:
    template: Pytree (nested mappings) with the naming the model expects.
    restored: Restored pytree, possibly with the other naming.

  Returns:
    `restored` with mapping keys renamed to follow `template`.
  """
  if not (isinstance(template, Mapping) and isinstance(restored, Mapping)):
    return restored
  by_canonical = collections.defaultdict(list)
  for r_key in restored:
    by_canonical[_strip_remat_prefix(r_key)].append(r_key)
  out = {}
  used = set()
  for t_key, t_val in template.items():
    if t_key in restored:
      r_key = t_key
    else:
      candidates = [
          k
          for k in by_canonical.get(_strip_remat_prefix(t_key), [])
          if k not in template
      ]
      if len(candidates) != 1:
        continue
      r_key = candidates[0]
    used.add(r_key)
    out[t_key] = _align_remat_names(t_val, restored[r_key])
  for r_key, r_val in restored.items():
    if r_key not in used:
      out[r_key] = r_val
  return out


def make_restore_args(
    target: Any, target_sharding: jax.sharding.Sharding
) -> Any:
  """Creates concrete ArrayRestoreArgs with target_sharding for every array

  leaf in target.

  Args:
    target: Target pytree (e.g. TrainState, dict of params, array).
    target_sharding: Target concrete jax.sharding.Sharding (e.g. NamedSharding).

  Returns:
    A pytree matching target with ArrayRestoreArgs for array leaves and
    RestoreArgs for scalar/string leaves.
  """

  def _to_abstract_or_self(x: Any) -> Any:
    if isinstance(x, (jax.Array, np.ndarray, jax.ShapeDtypeStruct)):
      return jax.ShapeDtypeStruct(x.shape, x.dtype)
    return x

  abstract_target = jax.tree.map(_to_abstract_or_self, target)
  sharding_tree = jax.tree.map(
      lambda x: target_sharding
      if isinstance(x, jax.ShapeDtypeStruct)
      else None,
      abstract_target,
  )
  return ocp.checkpoint_utils.construct_restore_args(
      abstract_target, sharding_tree
  )


_make_restore_args = make_restore_args


def _restore_params(
    ckpt_dir: str,
    step: int,
    template_params: Any,
    use_ema: bool = True,
    target_sharding: jax.sharding.Sharding | None = None,
) -> Any:
  """Restores (ema_)params of a saved train_state, e.g. for autoguidance.

  The checkpoint may have been saved with a different `use_remat` setting than
  `template_params`; module names are aligned via `_align_remat_names`.

  Args:
    ckpt_dir: Checkpoint manager directory containing `<step>/train_state`.
    step: Step to restore; negative means the latest step in `ckpt_dir`.
    template_params: Params pytree giving the structure and dtypes.
    use_ema: Restore `ema_params` if True, else the raw `params`.
    target_sharding: Optional target concrete sharding for restored arrays.

  Returns:
    The restored params pytree.
  """
  checkpoint_dir = epath.Path(ckpt_dir)
  if (checkpoint_dir / 'train_state').exists():
    state_subdir = checkpoint_dir / 'train_state'
  else:
    if step < 0:
      manager = ocp.CheckpointManager(
          directory=checkpoint_dir,
          checkpointers={
              'train_state': ocp.Checkpointer(
                  ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
              ),
          },
      )
      latest = manager.latest_step()
      if latest is None:
        raise ValueError(f'No checkpoint found in {checkpoint_dir}')
      step = int(latest)
    state_subdir = checkpoint_dir / str(step) / 'train_state'
  key = 'ema_params' if use_ema else 'params'
  raw = None
  errors = []
  target_item = {key: template_params}
  target_restore_args = (
      make_restore_args(target_item, target_sharding)
      if target_sharding is not None
      else None
  )
  for handler in (
      ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True),
      ocp.PyTreeCheckpointHandler(),
  ):
    try:
      ckptr = ocp.Checkpointer(handler)
      if target_restore_args is not None:
        try:
          raw = ckptr.restore(
              state_subdir,
              item=target_item,
              restore_args=target_restore_args,
              partial_restore=True,
          )
        except Exception as e_part:  # pylint: disable=broad-except
          logging.warning(
              'Partial restore with target_sharding failed (%s); trying'
              ' unconstrained restore.',
              e_part,
          )
          raw = ckptr.restore(state_subdir)
      else:
        raw = ckptr.restore(state_subdir)
      break
    except Exception as e:  # pylint: disable=broad-except
      errors.append(e)
  if raw is None:
    raise ValueError(f'Could not restore {state_subdir}: {errors}')
  if key not in raw:
    raise ValueError(
        f'Key {key!r} not in restored state from {state_subdir}:'
        f' {list(raw.keys())}'
    )
  logging.info('Restored %s from %r step %d', key, checkpoint_dir, step)
  restored_val = _cast_params_like(template_params, raw[key])
  if target_sharding is not None:
    restored_val = jax.device_put(restored_val, target_sharding)
  return restored_val


def average_pytrees(trees: Sequence[Any], weights: Sequence[float]) -> Any:
  """Computes weighted average of a list of pytrees with identical structure

  and shapes.

  Args:
    trees: Sequence of pytrees.
    weights: Sequence of floats summing to 1.0.

  Returns:
    Pytree with weighted average leaves.

  Raises:
    ValueError: If weights do not sum to 1.0, lengths mismatch, or trees/shapes
      mismatch.
  """
  if len(trees) != len(weights):
    raise ValueError(
        f'Mismatch: len(trees)={len(trees)} != len(weights)={len(weights)}'
    )
  if not trees:
    raise ValueError('trees sequence cannot be empty')
  weights = [float(w) for w in weights]
  total_weight = sum(weights)
  if not np.isclose(total_weight, 1.0, atol=1e-5):
    raise ValueError(
        f'Weights must sum to 1.0 (within atol=1e-5), got sum={total_weight}:'
        f' {weights}'
    )

  ref_struct = jax.tree_util.tree_structure(trees[0])
  for i, tree in enumerate(trees[1:], start=1):
    if jax.tree_util.tree_structure(tree) != ref_struct:
      raise ValueError(
          f'Tree {i} structure does not match reference tree 0 structure:'
          f' {jax.tree_util.tree_structure(tree)} !='
          f' {ref_struct}'
      )

  def _leaf_avg(*leaves):
    ref_shape = leaves[0].shape
    ref_dtype = leaves[0].dtype
    for idx, leaf in enumerate(leaves[1:], start=1):
      if leaf.shape != ref_shape:
        raise ValueError(
            f'Leaf shape mismatch at tree {idx}: shape {leaf.shape} !='
            f' {ref_shape}'
        )
    acc = jnp.zeros_like(leaves[0], dtype=jnp.float32)
    for w, leaf in zip(weights, leaves):
      acc = acc + w * jnp.asarray(leaf, dtype=jnp.float32)
    return jnp.asarray(acc, dtype=ref_dtype)

  return jax.tree_util.tree_map(_leaf_avg, *trees)


def effective_polyak_decay(
    step: Any,
    decay: float,
    switch_step: int = 0,
    decay_after: float = 0.0,
) -> Any:
  """EMA decay to pass to `utils.update_state` at train step `step`.

  With `switch_step` <= 0 this returns `decay` unchanged (default path). With
  `switch_step` > 0, `decay` applies before `switch_step` and `decay_after`
  from `switch_step` on, with the (1 + s) / (10 + s) EMA warm-up restarted at
  s = step - switch_step. This reproduces a warm start that resets the step
  counter: sota_fm_02 (stage B) restored r2_05 @560k (EMA 0.999) and continued
  with EMA 0.9999 from train step 0. `update_state` further applies
  min(., (1 + step) / (10 + step)), which is >= the restarted warm-up, so the
  restarted warm-up is what takes effect.

  Args:
    step: Train step before the update (scalar, may be traced).
    decay: EMA decay before `switch_step`.
    switch_step: Step of the switch; <= 0 disables it.
    decay_after: EMA decay from `switch_step` on.

  Returns:
    `decay` (a Python float) if the switch is disabled, else a float32 scalar.
  """
  if switch_step <= 0:
    return decay
  s = jnp.asarray(step, jnp.float32) - float(switch_step)
  s_pos = jnp.maximum(s, 0.0)
  restarted = jnp.minimum(float(decay_after), (1.0 + s_pos) / (10.0 + s_pos))
  return jnp.where(s < 0.0, jnp.float32(decay), restarted)


def ema_update(ema: Any, new_params: Any, step: Any, decay: float) -> Any:
  """One Polyak EMA update exactly as in `utils.update_state`.

  Args:
    ema: Current EMA pytree.
    new_params: Params after the optimizer update of train step `step`.
    step: Train step before that update (`state.step - 1` after it).
    decay: Target EMA decay; warmed up as min(decay, (1 + step) / (10 + step)).

  Returns:
    Updated EMA pytree.
  """
  d = jnp.minimum(decay, (1.0 + step) / (10.0 + step))
  return jax.tree_util.tree_map(
      lambda old, new: d * old + (1 - d) * new, ema, new_params
  )


def uniform_soup(trees: Sequence[Any]) -> Any:
  """Uniform weight-space average (float32 accumulation, `average_pytrees`)."""
  return average_pytrees(trees, [1.0 / len(trees)] * len(trees))


def paired_draw_rngs(
    seed: int, n_draws: int, eval_step: int = 10
) -> list[jax.Array]:
  """Per-draw keys of the eval-only protocol (max_steps=10, 1+extra draws).

  Draw 0 is the headline generation at the first eval (`generation_rng` after
  one split of key(generation_seed)); draw k >= 1 is the `eval_extra_draws`
  key fold_in(fold_in(key(seed), 1_000_003 + k), eval_step). Scoring a model
  with these keys via `score_draw` reuses the prior noise and rotations of the
  eval-only waves with the same `generation_seed` (e.g. pfm22 seeds 1617 and
  1819), so the draws are paired with those runs.

  Args:
    seed: generation_seed of the eval-only run to pair with.
    n_draws: Number of draws (1 headline + n_draws - 1 extra draws).
    eval_step: Train step of the eval in the eval-only run (10).

  Returns:
    List of `n_draws` PRNG keys.
  """
  base = jax.random.key(seed)
  keys = [jax.random.split(base)[1]]
  for k in range(1, n_draws):
    keys.append(
        jax.random.fold_in(jax.random.fold_in(base, 1_000_003 + k), eval_step)
    )
  return keys


def _parse_soup_spec(
    ckpts_val: Any, weights_val: Any, steps_val: Any
) -> tuple[list[str], list[float], list[int]]:
  """Parses checkpoint paths, weights, and optional steps for model soup."""
  if not ckpts_val:
    return [], [], []
  if isinstance(ckpts_val, (list, tuple)):
    ckpts = [str(x).strip() for x in ckpts_val if str(x).strip()]
  else:
    s = str(ckpts_val).strip()
    if s.startswith('[') and s.endswith(']'):
      s = s[1:-1]
    ckpts = [x.strip() for x in re.split(r'[,+]', s) if x.strip()]

  if not weights_val:
    raise ValueError(
        'model_soup_weights must be provided when model_soup_ckpts is set:'
        f' {ckpts}'
    )
  if isinstance(weights_val, (list, tuple)):
    weights = [float(x) for x in weights_val]
  else:
    s = str(weights_val).strip()
    if s.startswith('[') and s.endswith(']'):
      s = s[1:-1]
    weights = [float(x.strip()) for x in re.split(r'[,+]', s) if x.strip()]

  if steps_val:
    if isinstance(steps_val, (list, tuple)):
      steps = [int(x) for x in steps_val]
    else:
      s = str(steps_val).strip()
      if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
      steps = [int(x.strip()) for x in re.split(r'[,+]', s) if x.strip()]
  else:
    steps = [-1] * len(ckpts)

  if len(ckpts) != len(weights):
    raise ValueError(
        f'Mismatch: {len(ckpts)} model_soup_ckpts != {len(weights)}'
        ' model_soup_weights'
    )
  if len(steps) != len(ckpts):
    raise ValueError(
        f'Mismatch: {len(steps)} model_soup_steps != {len(ckpts)}'
        ' model_soup_ckpts'
    )
  return ckpts, weights, steps


def restore_model_soup(
    ckpts: Sequence[str],
    weights: Sequence[float],
    template_params: Any,
    steps: Sequence[int] | None = None,
    target_sharding: jax.sharding.Sharding | None = None,
) -> Any:
  """Restores EMA params from multiple checkpoints and returns their weighted

  average.
  """
  if steps is None or not steps:
    steps = [-1] * len(ckpts)
  elif len(steps) != len(ckpts):
    raise ValueError(f'len(steps)={len(steps)} != len(ckpts)={len(ckpts)}')

  trees = []
  for ckpt_path, step in zip(ckpts, steps):
    p = epath.Path(ckpt_path)
    actual_path = str(p)
    actual_step = step
    if (p / 'train_state').exists() and p.name.isdigit():
      actual_path = str(p.parent)
      actual_step = int(p.name)
    elif (p / 'train_state').exists():
      actual_path = str(p)
      actual_step = -1
    ema_p = _restore_params(
        ckpt_dir=actual_path,
        step=actual_step,
        template_params=template_params,
        use_ema=True,
        target_sharding=target_sharding,
    )
    trees.append(ema_p)

  return average_pytrees(trees, weights)


def _cast_params_like(template: Any, restored: Any) -> Any:
  """Casts `restored` onto `template`'s structure/dtypes (remat-agnostic)."""
  return jax.tree_util.tree_map(
      lambda t, r: jnp.asarray(r, dtype=t.dtype),
      template,
      _align_remat_names(template, restored),
  )


# Keys of an `eval_variants` entry -> (sampler field, type).
_VARIANT_KEYS = {
    'w': ('w', float),  # autoguidance weight
    'tmin': ('tmin', float),  # autoguidance interval start
    'tmax': ('tmax', float),  # autoguidance interval end
    't_min': ('tmin', float),  # alias for tmin
    't_max': ('tmax', float),  # alias for tmax
    'steps': ('steps', int),  # sample_steps
    'step': ('steps', int),  # alias for steps
    'n_steps': ('steps', int),  # alias for steps
    'solver': ('solver', str),  # sample_solver
    'sched': ('sched', str),  # sample_schedule
    'schedule': ('sched', str),  # alias for sched
    'temp': ('temp', float),  # prior noise std (sampling temperature)
    'churn': ('churn', float),  # stochastic churn factor
    'ctmin': ('ctmin', float),  # churn time window start
    'ctmax': ('ctmax', float),  # churn time window end
    'churn_tmin': ('ctmin', float),  # alias for ctmin
    'churn_tmax': ('ctmax', float),  # alias for ctmax
    'ds': ('ds', lambda v: v.lower() in ('true', '1', 't', 'yes')),
    'decouple_scale': ('ds', lambda v: v.lower() in ('true', '1', 't', 'yes')),
}
_AG_VARIANT_KEYS = (
    'w',
    'tmin',
    'tmax',
    't_min',
    't_max',
    'ds',
    'decouple_scale',
)


def parse_eval_variants(
    spec: str, has_autoguidance: bool = True
) -> list[tuple[str, dict[str, Any]]]:
  """Parses `config.eval_variants` into (name, sampler overrides) pairs.

  Supports both key-value specifications separated by '+' (e.g.
  'tmin=0.2+w=2.0:tmax=0.7') and positional/named specifications separated
  by ',' or '+' (e.g.
  'ag150_t80:euler:100:0.0:1.5:0.0:0.8,ag165_t75:euler:100:0.0:1.65:0.0:0.75').

  Args:
    spec: The variant specification; '' means no variants.
    has_autoguidance: Whether an autoguidance model is configured; guidance keys
      (w, tmin, tmax) require one.

  Returns:
    List of (name, overrides) pairs.

  Raises:
    ValueError: On malformed specs, unknown keys, duplicate names or guidance
      keys without an autoguidance model.
  """
  out = []
  variants = [v.strip() for v in re.split(r'[,+]', spec) if v.strip()]
  for variant in variants:
    parts = variant.split(':')
    if '=' not in variant and len(parts) in (7, 8):
      if len(parts) == 7:
        v_name, v_solver, v_steps, v_sched, v_w, v_tmin, v_tmax = parts
        v_ds = None
      else:
        v_name, v_solver, v_steps, v_sched, v_w, v_tmin, v_tmax, v_ds_str = (
            parts
        )
        v_ds = v_ds_str.lower() in ('true', '1', 't', 'yes', 'ds')
      w_val = float(v_w)
      if not has_autoguidance and w_val != 1.0:
        raise ValueError(
            f'Eval variant {variant!r} needs autoguidance_ckpt_path: {spec!r}'
        )
      overrides = {
          'solver': v_solver,
          'steps': int(v_steps),
          'w': w_val,
          'tmin': float(v_tmin),
          'tmax': float(v_tmax),
      }
      if v_ds is not None:
        overrides['ds'] = v_ds
      if v_sched not in ('0.0', '0'):
        overrides['sched'] = v_sched
      if v_name in {n for n, _ in out}:
        raise ValueError(f'Duplicate eval variant {v_name!r} in {spec!r}')
      out.append((v_name, overrides))
      continue

    overrides = {}
    for item in variant.split(':'):
      if '=' not in item:
        raise ValueError(f'Malformed eval variant item {item!r} in {spec!r}')
      key, value = item.split('=', 1)
      if key not in _VARIANT_KEYS:
        raise ValueError(f'Unknown eval variant key {key!r} in {spec!r}')
      if key in _AG_VARIANT_KEYS and not has_autoguidance:
        raise ValueError(
            f'Eval variant key {key!r} needs autoguidance_ckpt_path: {spec!r}'
        )
      field, cast = _VARIANT_KEYS[key]
      overrides[field] = cast(value)
    name = variant.replace('=', '').replace(':', '_')
    if name in {n for n, _ in out}:
      raise ValueError(f'Duplicate eval variant {name!r} in {spec!r}')
    out.append((name, overrides))
  return out


def guard_16_gpu(expected: int = 16, enabled: bool = True) -> None:
  """Verifies device count matches expected GPU count (fails fast on degraded

  nodes).
  """
  if not enabled:
    return
  n_dev = len(jax.devices())
  if n_dev != expected:
    raise RuntimeError(
        f'16-GPU Guard: Expected {expected} devices but found {n_dev}.'
        ' Failing fast to trigger Borg reschedule on healthy node.'
    )


def train_flow_matching(
    config: ml_collections.ConfigDict,
    log_dir: str,
    ckpt_workdir: str | None = None,
    ckpt_dir_override: str | None = None,
    ckpt_root_dir: str | None = None,
) -> None | jax.Array:
  """Trains a flow matching model.

  Args:
    config: Configuration for training.
    log_dir: Directory to write logs.
    ckpt_workdir: Optional directory to store heavy checkpoints locally.
    ckpt_dir_override: Direct directory to store checkpoints.
    ckpt_root_dir: Root directory for same-cell checkpoints.

  Returns:
    Generated samples after training.
  """
  if config.get('guard_16_gpu', False):
    guard_16_gpu(
        expected=int(config.get('num_devices', 16)),
        enabled=bool(config.guard_16_gpu),
    )

  workdir: epath.Path = epath.Path(config.workdir) / f'{config.name_str}/'
  log_path: epath.Path = epath.Path(log_dir) / f'{config.name_str}/'

  flag_ckpt_workdir = (
      getattr(flags.FLAGS, 'ckpt_workdir', None)
      if hasattr(flags, 'FLAGS')
      else None
  )
  flag_ckpt_dir_override = (
      getattr(flags.FLAGS, 'ckpt_dir_override', None)
      if hasattr(flags, 'FLAGS')
      else None
  )
  flag_ckpt_root_dir = (
      getattr(flags.FLAGS, 'ckpt_root_dir', None)
      if hasattr(flags, 'FLAGS')
      else None
  )

  raw_direct = (
      ckpt_dir_override
      or flag_ckpt_dir_override
      or config.get('ckpt_dir_override', None)
  )
  raw_workdir = (
      ckpt_workdir or flag_ckpt_workdir or config.get('ckpt_workdir', None)
  )
  raw_root_dir = (
      ckpt_root_dir or flag_ckpt_root_dir or config.get('ckpt_root_dir', None)
  )

  is_nf_cell = (
      os.environ.get('BORG_CELL', '').lower() == 'nf'
      or 'nf' in os.environ.get('BORG_CELL', '').lower()
      or str(config.get('cell', '')).lower() == 'nf'
  )
  str_workdir = str(workdir)
  raw_log_dir_str = str(log_dir)
  str_log_dir = str(log_path)

  if raw_direct:
    ckpt_dir: epath.Path = epath.Path(raw_direct)
  elif raw_workdir:
    ckpt_dir = epath.Path(raw_workdir) / f'{config.name_str}/'
  elif (
      raw_root_dir
      or str_log_dir.startswith('/cns/je-d/')
      or str_workdir.startswith('/cns/je-d/')
      or is_nf_cell
  ):
    root = raw_root_dir or '/cns/nf-d/home/riegerfr/point/fm_ckpts'
    xid_wid = None
    m = re.search(r'/xm/[^/]+/(\d+)/(\d+)', raw_log_dir_str)
    if m:
      xid_wid = f'{m.group(1)}_{m.group(2)}'
    else:
      m = re.search(r'/(\d{7,12})/(\d+)(?:/|$)', raw_log_dir_str)
      if m:
        xid_wid = f'{m.group(1)}_{m.group(2)}'
      elif os.environ.get('XM_EXPERIMENT_ID') and os.environ.get(
          'XM_WORK_UNIT_ID'
      ):
        xm_exp = os.environ.get('XM_EXPERIMENT_ID')
        xm_wu = os.environ.get('XM_WORK_UNIT_ID')
        xid_wid = f'{xm_exp}_{xm_wu}'

    if xid_wid:
      ckpt_dir = epath.Path(root) / xid_wid / f'{config.name_str}/'
    elif '/xm/point/' in raw_log_dir_str:
      rel_path = raw_log_dir_str.split('/xm/point/', 1)[1].strip('/')
      ckpt_dir = epath.Path(root) / rel_path / f'{config.name_str}/'
    elif '/point/' in str_workdir:
      rel_path = str_workdir.split('/point/', 1)[1].strip('/')
      ckpt_dir = epath.Path(root) / rel_path
    else:
      ckpt_dir = epath.Path(root) / f'{config.name_str}/'
  else:
    ckpt_dir = workdir

  logging.info(
      'workdir: %s, log_dir: %s, ckpt_dir: %s', workdir, log_path, ckpt_dir
  )
  logging.info('JAX version: %s', jax.__version__)
  logging.info('JAXlib version: %s', jax.lib.__version__)

  workdir.mkdir(parents=True, exist_ok=True)
  log_path.mkdir(parents=True, exist_ok=True)
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  writer = metric_writers.create_default_writer(
      log_path, just_logging=jax.process_index() > 0
  )
  logging.info('Starting training with config: %s', config)

  model_rng, train_rng, val_rng, data_rng = jax.random.split(
      jax.random.key(config.seed), num=4
  )

  train_dataloader, n_train_samples, val_dataloader, n_val_samples, _ = (
      _get_fake_dataloaders(
          per_device_batch_size=config.batch_size * config.n_combine_samples,
          n_points=config.get('n_points', 2048),
          n_feat=config.get('out_dim', 3),
          seed=int(jax.random.key_data(data_rng)[0]),
          num_devices=config.num_devices,
      )
  )
  logging.info(
      'n_train_samples %d, n_val_samples %d', n_train_samples, n_val_samples
  )
  logging.info('config.batch_size = %d', config.batch_size)
  logging.debug(
      'DEBUG: config.n_combine_samples = %d', config.n_combine_samples
  )
  logging.debug(
      'DEBUG: Effective batch_size for get_dataloaders = %d',
      config.batch_size * config.n_combine_samples,
  )
  train_iter = iter(train_dataloader)

  generation_rng = jax.random.key(config.generation_seed)
  first_batch = next(train_iter)
  logging.debug(
      "DEBUG: first_batch['coord'].shape = %s", first_batch['coord'].shape
  )
  init_coord, init_feat, init_cond = utils.prep_data(
      first_batch,
      config.coord_scale,
      config.feat_scale,
      config.n_points // config.n_combine_samples,
      n_combine_samples=config.n_combine_samples,
      use_feat=config.use_feat,
      cond_mode=config.cond_mode,
      dst_index_cond=config.get('dst_index_cond', False),
      add_dummy_cond=config.get('add_dummy_cond', False),
      # TODO(riegerfr): make dst_index_cond cleaner
      # (i.e. no dependency on train set string here)
  )
  logging.debug('DEBUG: init_coord.shape = %s', init_coord.shape)
  logging.debug(
      'DEBUG: config.batch_size for assertion = %d', config.batch_size
  )
  assert init_coord.shape[1] == config.n_points
  assert init_coord.shape[2] == 3
  expected_per_device_batch_size = (
      config.large_batch_size
      if (
          config.dynamic_batch_freq > 0
          or config.get('combined_batch_mode', False)
      )
      else config.batch_size
  )
  assert (
      init_coord.shape[0] // config.num_devices
      == expected_per_device_batch_size
  ), (
      f'n_train_samples was {n_train_samples} but init_coord.shape[0] is'
      f' {init_coord.shape[0]} and expected_per_device_batch_size is'
      f' {expected_per_device_batch_size} and config.num_devices is'
      f' {config.num_devices}'
  )

  if jax.process_index() == 0:
    # `storage.atomic_file` stages through a local tempfile, and the Borg task's
    # local scratch allowance is 64 MiB. The full init batch is
    # num_devices * large_batch_size * n_points * 3 * 4 bytes, which is 75 MB at
    # large_batch_size=48 and 101 MB at 64 -- both of which fail with ENOSPC
    # before training ever starts. This dump only exists for eyeballing the
    # input pipeline, so a few examples are enough.
    n_dump = min(int(init_coord.shape[0]), 32)
    with storage.atomic_file(
        str(workdir / 'init_data.npz'),
        'wb',  # local=False
    ) as f:
      save_dict = {'coord': np.asarray(init_coord[:n_dump])}
      if config.use_feat and init_feat is not None:
        save_dict['feat'] = np.asarray(init_feat[:n_dump])
      np.savez_compressed(f, **save_dict)

  feat_shape = init_feat.shape[2] if init_feat is not None else 0

  assert ((-1 < init_coord) & (init_coord < 1)).mean() > 0.9

  model, params, batch_stats = utils.get_model(
      model_rng,
      init_coord=init_coord[:1],
      init_feat=init_feat[:1] if init_feat is not None else None,
      config=config,
      cond=jnp.concat((init_cond[:1], jnp.zeros_like(init_cond[:1])), axis=1)
      if init_cond is not None
      else None,
      point_cond_mask=jnp.zeros((1, init_coord.shape[1]), dtype=bool)
      if config.point_cond > 0
      else None,
      n_combine_samples=config.n_combine_samples,
  )
  if config.optimizer in ('adamc', 'adamw_cosine'):
    warmup_steps = min(
        int(config.get('warmup_steps', 1000)), max(1, config.max_steps - 1)
    )
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=warmup_steps,
        decay_steps=config.max_steps,
    )
    if config.optimizer == 'adamc':
      wd = lambda c: config.wd * lr_sched(c) / config.lr
    else:
      wd = config.wd
    optimizer = utils.get_optimizer(config)(
        learning_rate=lr_sched, weight_decay=wd
    )
  elif config.optimizer == 'schedule_free_adamw':
    optimizer = utils.get_optimizer(config)(
        learning_rate=config.lr,
        weight_decay=config.wd,
        warmup_steps=config.get('warmup_steps', 0),
    )
  else:
    optimizer = utils.get_optimizer(config)(
        learning_rate=config.lr, weight_decay=config.wd
    )
  if config.clip_grad > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_grad), optimizer
    )

  best_x_gen = None

  state = utils.TrainState(
      step=0,
      params=params,
      ema_params=params,
      batch_stats=batch_stats,  # pyrefly: ignore[bad-argument-type]
      opt_state=optimizer.init(params),
      min_s_mmd_train=float('inf'),
  )

  mesh = sharding.Mesh(np.array(jax.devices()), ('batch',))
  batch_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec('batch'))
  replicate_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec())
  logging.info('Device mesh: %r', mesh)
  global_batch_size = config.batch_size * config.num_devices

  state = jax.device_put(state, replicate_sharding)

  if jax.process_index() == 0:
    for target_dir in [workdir, log_path]:
      target_dir.mkdir(parents=True, exist_ok=True)
      with storage.atomic_file(
          str(target_dir / 'init_coord.html'),
          'w',  # local=False
      ) as f:
        utils.plot_point_clouds(
            init_coord[:16], n_combine_samples=config.n_combine_samples
        ).write_html(f)

  async_options = ocp.AsyncOptions(timeout_secs=7200)
  latest_checkpoint_manager = ocp.CheckpointManager(
      directory=ckpt_dir / 'checkpoints',
      checkpointers={
          'train_state': ocp.AsyncCheckpointer(
              ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True),
              timeout_secs=7200,
              async_options=async_options,
          ),
          # pyrefly: ignore[bad-argument-type]
          'train_iter': ocp.Checkpointer(grain.OrbaxCheckpointHandler()),
      },
      options=ocp.CheckpointManagerOptions(
          max_to_keep=3,
          cleanup_tmp_directories=True,
          async_options=async_options,
      ),
  )  # to restore after preemption
  best_checkpoint_manager = ocp.CheckpointManager(
      directory=ckpt_dir / 'best_checkpoints',
      checkpointers={
          'train_state': ocp.AsyncCheckpointer(
              ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True),
              timeout_secs=7200,
              async_options=async_options,
          ),
      },
      options=ocp.CheckpointManagerOptions(
          max_to_keep=3,
          cleanup_tmp_directories=True,
          async_options=async_options,
      ),
  )  # for inference, workaround: https://github.com/google/orbax/issues/526

  if (
      latest_checkpoint_manager.latest_step() is None
      and (workdir / 'checkpoints').exists()
  ):
    legacy_mgr = ocp.CheckpointManager(
        directory=workdir / 'checkpoints',
        checkpointers={
            'train_state': ocp.AsyncCheckpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True),
                timeout_secs=7200,
                async_options=async_options,
            ),
            # pytype:disable=wrong-arg-types
            'train_iter': ocp.Checkpointer(grain.OrbaxCheckpointHandler()),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=3,
            cleanup_tmp_directories=True,
            async_options=async_options,
        ),
    )
    if legacy_mgr.latest_step():
      logging.info('Found legacy checkpoint in %s', workdir / 'checkpoints')
      latest_checkpoint_manager = legacy_mgr

  # True only after a warm start from init_ckpt_path (not an own-dir resume).
  warm_started = False
  if latest_checkpoint_manager.latest_step():
    if config.get('guard_16_gpu', False):
      guard_16_gpu(
          expected=int(config.get('num_devices', 16)),
          enabled=bool(config.guard_16_gpu),
      )
    restore_args = make_restore_args(state, replicate_sharding)
    restored_data = latest_checkpoint_manager.restore(
        latest_checkpoint_manager.latest_step(),
        items={'train_state': state, 'train_iter': train_iter},
        restore_kwargs={'train_state': {'restore_args': restore_args}},
    )
    state = restored_data['train_state']
    train_iter = restored_data['train_iter']
    logging.info(
        'Restored checkpoint for step %d',
        latest_checkpoint_manager.latest_step(),
    )
  else:
    logging.info('Starting training from scratch.')
    init_ckpt = config.get('init_ckpt_path', '') or config.get(
        'initial_checkpoint_path', ''
    )
    if init_ckpt:
      checkpoint_dir = epath.Path(init_ckpt)
      logging.info('initial checkpoint provided: %r', init_ckpt)
      initial_checkpoint_manager = ocp.CheckpointManager(
          directory=checkpoint_dir,
          checkpointers={
              'train_state': ocp.AsyncCheckpointer(
                  ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
              ),
          },
      )
      # initial_checkpoint_step / init_ckpt_step < 0 (default): latest step,
      # as before.
      raw_step = config.get('init_ckpt_step', None)
      if raw_step is None:
        raw_step = config.get('initial_checkpoint_step', -1)
      pinned_step = int(raw_step)
      if pinned_step >= 0 and pinned_step not in list(
          initial_checkpoint_manager.all_steps()
      ):
        raise ValueError(
            f'initial_checkpoint_step={pinned_step} not in'
            f' {initial_checkpoint_manager.all_steps()} of {checkpoint_dir}'
        )
      if initial_checkpoint_manager.latest_step() is not None:
        step_to_restore = (
            pinned_step
            if pinned_step >= 0
            else initial_checkpoint_manager.latest_step()
        )
        logging.info(
            'Loading initial weights from %r step %d',
            checkpoint_dir,
            step_to_restore,
        )
        state_subdir = checkpoint_dir / str(step_to_restore) / 'train_state'
        restored_state = None
        restore_args = make_restore_args(state, replicate_sharding)
        if state_subdir.exists():
          for handler in (
              ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True),
              ocp.PyTreeCheckpointHandler(),
          ):
            try:
              ckptr = ocp.Checkpointer(handler)
              if config.get('load_optimizer_state', True):
                restored_state = ckptr.restore(
                    state_subdir, item=state, restore_args=restore_args
                )
              else:
                try:
                  restored_state = ckptr.restore(
                      state_subdir, item=state, restore_args=restore_args
                  )
                except Exception:  # pylint: disable=broad-except
                  raw_restore_args = make_restore_args(
                      {'params': state.params, 'ema_params': state.ema_params},
                      replicate_sharding,
                  )
                  try:
                    raw = ckptr.restore(
                        state_subdir,
                        item={
                            'params': state.params,
                            'ema_params': state.ema_params,
                        },
                        restore_args=raw_restore_args,
                        partial_restore=True,
                    )
                  except Exception:  # pylint: disable=broad-except
                    raw = ckptr.restore(state_subdir)
                  # pyrefly: ignore[missing-attribute]
                  restored_state = state.replace(
                      params=_cast_params_like(state.params, raw['params']),
                      ema_params=_cast_params_like(
                          state.ema_params, raw['ema_params']
                      ),
                  )
              break
            except Exception as e_sub:  # pylint: disable=broad-except
              logging.warning(
                  'Direct train_state subdir restore with %r failed: %s',
                  handler,
                  e_sub,
              )
        if restored_state is None:
          try:
            restored_state = initial_checkpoint_manager.restore(
                step_to_restore,
                items={'train_state': state},
                restore_kwargs={'train_state': {'restore_args': restore_args}},
            )['train_state']
          except Exception as e:  # pylint: disable=broad-except
            logging.warning(
                'OCDBT/zarr3 restore failed (%s); falling back to legacy'
                ' PyTreeCheckpointHandler()',
                e,
            )
            legacy_mgr = ocp.CheckpointManager(
                directory=checkpoint_dir,
                checkpointers={
                    'train_state': ocp.AsyncCheckpointer(
                        ocp.PyTreeCheckpointHandler()
                    ),
                },
            )
            try:
              restored_state = legacy_mgr.restore(
                  step_to_restore,
                  items={'train_state': state},
                  restore_kwargs={
                      'train_state': {'restore_args': restore_args}
                  },
              )['train_state']
            except Exception as e2:  # pylint: disable=broad-except
              if config.get('load_optimizer_state', True):
                raise
              logging.warning(
                  'Structured restore failed with load_optimizer_state=False'
                  ' (%s); restoring raw dict and extracting params/ema_params.',
                  e2,
              )
              try:
                raw = initial_checkpoint_manager.restore(
                    step_to_restore,
                    items={'train_state': state},
                    restore_kwargs={
                        'train_state': {'restore_args': restore_args}
                    },
                )['train_state']
              except Exception:  # pylint: disable=broad-except
                raw = legacy_mgr.restore(
                    step_to_restore,
                    items={'train_state': state},
                    restore_kwargs={
                        'train_state': {'restore_args': restore_args}
                    },
                )['train_state']
              # pyrefly: ignore[missing-attribute]
              restored_state = state.replace(
                  params=_cast_params_like(state.params, raw['params']),
                  ema_params=_cast_params_like(
                      state.ema_params, raw['ema_params']
                  ),
              )
        continue_step = config.get('continue_step', False) or bool(
            config.get('init_ckpt_path', '')
        )
        new_step = step_to_restore if continue_step else 0
        if config.get('load_optimizer_state', True):
          state = state.replace(  # pyrefly: ignore[missing-attribute]
              step=new_step,
              params=restored_state.params,
              ema_params=restored_state.ema_params,
              batch_stats=restored_state.batch_stats,
              opt_state=restored_state.opt_state,
              min_s_mmd_train=float('inf'),
          )
        else:
          state = state.replace(  # pyrefly: ignore[missing-attribute]
              step=new_step,
              params=restored_state.params,
              ema_params=restored_state.ema_params,
              batch_stats=restored_state.batch_stats,
              min_s_mmd_train=float('inf'),
          )
        logging.info(
            'Restored initial weights from %r step %d (training starts at'
            ' step %d)',
            checkpoint_dir,
            step_to_restore,
            new_step,
        )
        warm_started = True
      else:
        logging.warning(
            'initial checkpoint %r specified, but step %d not found in'
            ' %r. Checkpoint steps: %r',
            init_ckpt,
            initial_checkpoint_manager.latest_step(),
            checkpoint_dir,
            initial_checkpoint_manager.all_steps(),
        )
      train_iter = iter(train_dataloader)

    soup_ckpts_raw = (
        getattr(flags.FLAGS, 'model_soup_ckpts', None)
        if hasattr(flags, 'FLAGS')
        and getattr(flags.FLAGS, 'model_soup_ckpts', None)
        else config.get('model_soup_ckpts', '')
    )
    soup_weights_raw = (
        getattr(flags.FLAGS, 'model_soup_weights', None)
        if hasattr(flags, 'FLAGS')
        and getattr(flags.FLAGS, 'model_soup_weights', None)
        else config.get('model_soup_weights', '')
    )
    soup_steps_raw = (
        getattr(flags.FLAGS, 'model_soup_steps', None)
        if hasattr(flags, 'FLAGS')
        and getattr(flags.FLAGS, 'model_soup_steps', None)
        else config.get('model_soup_steps', '')
    )
    soup_ckpts, soup_weights, soup_steps = _parse_soup_spec(
        soup_ckpts_raw, soup_weights_raw, soup_steps_raw
    )
    if soup_ckpts:
      logging.info(
          'Model soup active: restoring and averaging EMA params across %d'
          ' checkpoints with weights %s: %s',
          len(soup_ckpts),
          soup_weights,
          list(zip(soup_ckpts, soup_steps)),
      )
      soup_ema = restore_model_soup(
          soup_ckpts,
          soup_weights,
          template_params=state.ema_params,
          steps=soup_steps,
          target_sharding=replicate_sharding,
      )
      state = state.replace(  # pyrefly: ignore[missing-attribute]
          params=soup_ema,
          ema_params=soup_ema,
          min_s_mmd_train=float('inf'),
      )
      warm_started = True

    if jax.process_index() == 0:
      with tf.io.gfile.GFile(
          tf.io.gfile.join(workdir, 'config.json'), 'w'
      ) as f:
        f.write(config.to_json_best_effort() + '\n')

    total_params = sum(np.prod(x.shape) for x in jax.tree.leaves(state.params))
    logging.info('Total parameters: %.4fM', total_params / 1e6)
    writer.write_scalars(0, {'total_params': total_params})
    writer.write_hparams({
        k: v
        for k, v in config.items()
        if isinstance(v, (bool, float, int, str))
    })
    writer.write_texts(0, {'config': str(config)})  # for easier matching
    # TODO(riegerfr): also log cl/commit number+ timestamp from xmanager

  # Ensure state is placed on replicate_sharding after any restore or soup
  state = jax.device_put(state, replicate_sharding)

  # Eval-time autoguidance (Karras et al., 2024): a "bad" model of the same
  # architecture (e.g. an early checkpoint) restored alongside the main model.
  # With `autoguidance_self_step` > 0 and no usable `autoguidance_ckpt_path`,
  # the run guides itself with a snapshot of its own EMA params taken at that
  # train step (an earlier, less-trained version of the same architecture),
  # persisted under `<ckpt_dir>/ag_self` so that restarts keep the same guide.
  # Before the snapshot exists, evals run without autoguidance (`ag_active`=0)
  # and guidance-only eval variants are skipped.
  autoguidance_params = None
  ag_self_step = int(config.get('autoguidance_self_step', 0))
  if config.get('autoguidance_ckpt_path', ''):
    try:
      autoguidance_params = jax.device_put(
          _restore_params(
              config.autoguidance_ckpt_path,
              int(config.get('autoguidance_ckpt_step', -1)),
              state.ema_params,
              use_ema=config.get('autoguidance_use_ema', True),
              target_sharding=replicate_sharding,
          ),
          replicate_sharding,
      )
    except Exception as e:  # pylint: disable=broad-except
      if ag_self_step <= 0:
        # Fail fast: silently disabling guidance trains a different experiment
        # than the one configured (pfm3 b8/b16, pfm4 b8/b16 on 2026-10-02).
        raise ValueError(
            'Could not restore autoguidance params from'
            f' {config.autoguidance_ckpt_path!r} (architecture mismatch?). Set'
            ' autoguidance_self_step > 0 to guide with an own early snapshot'
            ' instead.'
        ) from e
      logging.warning(
          'Could not restore autoguidance params (%s); falling back to'
          ' self-autoguidance at step %d.',
          e,
          ag_self_step,
      )
      autoguidance_params = None
  ag_self_manager = None
  if autoguidance_params is None and ag_self_step > 0:
    ag_self_manager = ocp.CheckpointManager(
        directory=ckpt_dir / 'ag_self',
        checkpointers={
            'train_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, cleanup_tmp_directories=True
        ),
    )
    if ag_self_manager.latest_step() is not None:
      autoguidance_params = jax.device_put(
          _restore_params(
              str(ckpt_dir / 'ag_self'),
              -1,
              state.ema_params,
              use_ema=True,
              target_sharding=replicate_sharding,
          ),
          replicate_sharding,
      )
      logging.info(
          'Restored self-autoguidance snapshot (step %d).',
          ag_self_manager.latest_step(),
      )

  # Optional dedicated EMA for the self-autoguidance snapshot (off by default:
  # the snapshot copies the main EMA). With `autoguidance_self_ema_decay` > 0 an
  # extra EMA with that decay is tracked from step 0 and copied at
  # `autoguidance_self_step`, e.g. EMA 0.9999 @100k while the main EMA is 0.999
  # (the headline guide r2_05_ema9999 @100k came from a 0.9999-EMA run). It is
  # saved next to every train-state checkpoint until the snapshot exists.
  guide_ema_decay = float(config.get('autoguidance_self_ema_decay', 0.0))
  guide_ema = None
  guide_ema_manager = None
  p_guide_ema_update = None
  if guide_ema_decay > 0.0:
    if ag_self_manager is None:
      raise ValueError(
          'autoguidance_self_ema_decay needs autoguidance_self_step > 0 and no'
          ' autoguidance_ckpt_path'
      )
    if config.get('combined_batch_mode', False):
      raise ValueError(
          'autoguidance_self_ema_decay is not supported with'
          ' combined_batch_mode'
      )
    if autoguidance_params is None:
      guide_ema_manager = ocp.CheckpointManager(
          directory=ckpt_dir / 'guide_ema',
          checkpointers={
              'guide_ema': ocp.Checkpointer(
                  ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
              ),
          },
          options=ocp.CheckpointManagerOptions(
              max_to_keep=3, cleanup_tmp_directories=True
          ),
      )
      cur_step = int(jax.device_get(state.step))
      if cur_step == 0:
        guide_ema = jax.device_put(
            jax.tree_util.tree_map(jnp.copy, state.ema_params),
            replicate_sharding,
        )
      elif cur_step in list(guide_ema_manager.all_steps()):
        guide_restore_args = make_restore_args(
            state.ema_params, replicate_sharding
        )
        guide_ema = jax.device_put(
            guide_ema_manager.restore(
                cur_step,
                items={'guide_ema': state.ema_params},
                restore_kwargs={
                    'guide_ema': {'restore_args': guide_restore_args}
                },
            )['guide_ema'],
            replicate_sharding,
        )
        logging.info('Restored guide EMA at step %d.', cur_step)
      else:
        init_guide_ema = None
        candidate_bases = []
        for p_key in ('init_ckpt_path', 'initial_checkpoint_path'):
          val = str(config.get(p_key, '') or '')
          if val:
            candidate_bases.extend([epath.Path(val).parent, epath.Path(val)])
        for candidate_base in candidate_bases:
          candidate_dir = candidate_base / 'guide_ema'
          if candidate_dir.exists():
            init_mgr = ocp.CheckpointManager(
                directory=candidate_dir,
                checkpointers={
                    'guide_ema': ocp.Checkpointer(
                        ocp.PyTreeCheckpointHandler(
                            use_ocdbt=True, use_zarr3=True
                        )
                    ),
                },
                options=ocp.CheckpointManagerOptions(max_to_keep=3),
            )
            avail_steps = list(init_mgr.all_steps())
            guide_restore_args = make_restore_args(
                state.ema_params, replicate_sharding
            )
            if cur_step in avail_steps:
              init_guide_ema = init_mgr.restore(
                  cur_step,
                  items={'guide_ema': state.ema_params},
                  restore_kwargs={
                      'guide_ema': {'restore_args': guide_restore_args}
                  },
              )['guide_ema']
              logging.info(
                  'Restored guide EMA at step %d from %s.',
                  cur_step,
                  candidate_dir,
              )
              break
            elif avail_steps:
              closest = max(
                  [s for s in avail_steps if s <= cur_step],
                  default=max(avail_steps),
              )
              init_guide_ema = init_mgr.restore(
                  closest,
                  items={'guide_ema': state.ema_params},
                  restore_kwargs={
                      'guide_ema': {'restore_args': guide_restore_args}
                  },
              )['guide_ema']
              logging.info(
                  'Restored guide EMA from closest step %d (target %d)'
                  ' from %s.',
                  closest,
                  cur_step,
                  candidate_dir,
              )
              break
        if init_guide_ema is not None:
          guide_ema = jax.device_put(init_guide_ema, replicate_sharding)
          guide_ema_manager.save(cur_step, items={'guide_ema': guide_ema})
          guide_ema_manager.wait_until_finished()
        else:
          raise ValueError(
              f'No guide EMA saved at resume step {cur_step} in'
              f' {ckpt_dir / "guide_ema"} (steps'
              f' {list(guide_ema_manager.all_steps())}).'
          )

      def _guide_ema_step(ema, new_params, step_before):
        return ema_update(ema, new_params, step_before, guide_ema_decay)

      p_guide_ema_update = jax.jit(
          _guide_ema_step,
          in_shardings=(
              replicate_sharding,
              replicate_sharding,
              replicate_sharding,
          ),
          out_shardings=replicate_sharding,
      )

  train_start_time = time.time()
  running_train_step = 0
  running_train_loss = 0.0
  running_train_loss_squared = 0.0  # For variance calculation
  running_aux_loss = 0.0
  log_dict = {
      'sum': collections.defaultdict(float),
      'count': collections.defaultdict(int),
  }

  extra_ema_decays = [
      float(x.strip())
      for x in str(config.get('polyak_decays_extra', '')).split(',')
      if x.strip()
  ]
  extra_emas = {
      f'ema_{d}': jax.device_put(
          jax.tree_util.tree_map(jnp.copy, state.ema_params),
          replicate_sharding,
      )
      for d in extra_ema_decays
  }
  extra_ema_manager = None
  if extra_ema_decays:
    extra_ema_manager = ocp.CheckpointManager(
        directory=ckpt_dir / 'extra_emas',
        checkpointers={
            'extra_emas': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=3, cleanup_tmp_directories=True
        ),
    )
    if extra_ema_manager.latest_step() is not None:
      extra_restore_args = make_restore_args(extra_emas, replicate_sharding)
      restored_extra = extra_ema_manager.restore(
          extra_ema_manager.latest_step(),
          items={'extra_emas': extra_emas},
          restore_kwargs={'extra_emas': {'restore_args': extra_restore_args}},
      )
      extra_emas = jax.device_put(
          restored_extra['extra_emas'], replicate_sharding
      )
      logging.info(
          'Restored extra EMAs from step %d',
          extra_ema_manager.latest_step(),
      )

  def _update_extra_emas_step(cur_extra_emas, new_params, cur_step):
    decay_warmup = (1.0 + cur_step) / (10.0 + cur_step)
    out = {}
    for d in extra_ema_decays:
      key = f'ema_{d}'
      d_val = float(d)
      d_up = jnp.where(d_val >= 1.0, 1.0, jnp.minimum(d_val, decay_warmup))
      out[key] = jax.tree_util.tree_map(
          lambda old, new: jnp.where(
              d_val >= 1.0, old, d_up * old + (1.0 - d_up) * new
          ),
          cur_extra_emas[key],
          new_params,
      )
    return out

  if extra_ema_decays:
    p_update_extra_emas = jax.jit(
        _update_extra_emas_step,
        in_shardings=(
            replicate_sharding,
            replicate_sharding,
            replicate_sharding,
        ),
        out_shardings=replicate_sharding,
    )
  else:
    p_update_extra_emas = None

  posthoc_ema = bool(config.get('posthoc_ema', False))
  posthoc_emas = None
  p_update_posthoc_emas = None
  posthoc_ema_manager = None
  if posthoc_ema:
    posthoc_gamma1 = float(config.get('posthoc_gamma1', 6.94))
    posthoc_gamma2 = float(config.get('posthoc_gamma2', 16.97))
    posthoc_emas = {
        'ema_gamma1': jax.device_put(
            jax.tree_util.tree_map(
                lambda x: jnp.copy(x).astype(jnp.float32), state.ema_params
            ),
            replicate_sharding,
        ),
        'ema_gamma2': jax.device_put(
            jax.tree_util.tree_map(
                lambda x: jnp.copy(x).astype(jnp.float32), state.ema_params
            ),
            replicate_sharding,
        ),
    }
    posthoc_ema_manager = ocp.CheckpointManager(
        directory=ckpt_dir / 'posthoc_emas',
        checkpointers={
            'posthoc_emas': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=3, cleanup_tmp_directories=True
        ),
    )
    if posthoc_ema_manager.latest_step() is not None:
      ph_restore_args = make_restore_args(posthoc_emas, replicate_sharding)
      restored_ph = posthoc_ema_manager.restore(
          posthoc_ema_manager.latest_step(),
          items={'posthoc_emas': posthoc_emas},
          restore_kwargs={'posthoc_emas': {'restore_args': ph_restore_args}},
      )
      posthoc_emas = jax.device_put(
          restored_ph['posthoc_emas'], replicate_sharding
      )
      logging.info(
          'Restored post-hoc EMAs from step %d',
          posthoc_ema_manager.latest_step(),
      )

    def _update_posthoc_emas_step(cur_posthoc_emas, new_params, cur_step):
      t = jnp.maximum(cur_step.astype(jnp.float32), 0.0)
      beta1 = jnp.where(
          t <= 0.0, 0.0, (t / (t + 1.0)) ** (posthoc_gamma1 + 1.0)
      )
      beta2 = jnp.where(
          t <= 0.0, 0.0, (t / (t + 1.0)) ** (posthoc_gamma2 + 1.0)
      )
      out = {}
      out['ema_gamma1'] = jax.tree_util.tree_map(
          lambda old, new: beta1 * old.astype(jnp.float32)
          + (1.0 - beta1) * new.astype(jnp.float32),
          cur_posthoc_emas['ema_gamma1'],
          new_params,
      )
      out['ema_gamma2'] = jax.tree_util.tree_map(
          lambda old, new: beta2 * old.astype(jnp.float32)
          + (1.0 - beta2) * new.astype(jnp.float32),
          cur_posthoc_emas['ema_gamma2'],
          new_params,
      )
      return out

    p_update_posthoc_emas = jax.jit(
        _update_posthoc_emas_step,
        in_shardings=(
            replicate_sharding,
            replicate_sharding,
            replicate_sharding,
        ),
        out_shardings={
            'ema_gamma1': replicate_sharding,
            'ema_gamma2': replicate_sharding,
        },
    )

  fixed_t_eval_active = bool(
      config.get('eval_fixed_t', False) or config.get('posthoc_ema', False)
  )
  fixed_val_batch: Any = None
  fixed_train_batch: Any = None
  p_eval_fixed_t: Any = None
  p_solve_posthoc: Any = None

  if fixed_t_eval_active:

    def _collect_fixed_data(dataloader, n_target=64):
      batches_x1, batches_cond = [], []
      cur = 0
      for b in dataloader:
        c, f, cond_arr = utils.prep_data(
            b,
            config.coord_scale,
            config.feat_scale,
            config.n_points // config.n_combine_samples,
            n_combine_samples=config.n_combine_samples,
            use_feat=config.use_feat,
            cond_mode=config.cond_mode,
            dst_index_cond=config.get('dst_index_cond', False),
            add_dummy_cond=config.get('add_dummy_cond', False),
        )
        if config.get('use_bf16', False):
          c = c.astype(jnp.bfloat16)
          if f is not None:
            f = f.astype(jnp.bfloat16)
        x_1 = jnp.concatenate((c, f), axis=2) if f is not None else c
        batches_x1.append(x_1)
        if cond_arr is not None:
          batches_cond.append(cond_arr)
        cur += x_1.shape[0]
        if cur >= n_target:
          break
      all_x1 = jnp.concatenate(batches_x1, axis=0)
      n_actual = min(all_x1.shape[0], n_target)
      if n_actual % config.num_devices != 0:
        n_actual = max(
            config.num_devices,
            (n_actual // config.num_devices) * config.num_devices,
        )
      out_x1 = all_x1[:n_actual]
      if batches_cond:
        raw_cond = jnp.concatenate(batches_cond, axis=0)[:n_actual]
        if float(config.get('feat_cond_dropout_threshold', 0.0)) > 0.0:
          out_cond = jnp.concatenate(
              (raw_cond, jnp.ones_like(raw_cond)), axis=-1
          )
        else:
          out_cond = jnp.concatenate(
              (jnp.zeros_like(raw_cond), jnp.zeros_like(raw_cond)), axis=-1
          )
      else:
        out_cond = None
      return out_x1, out_cond

    val_x1, val_cond = _collect_fixed_data(val_dataloader, 64)
    train_x1, train_cond = _collect_fixed_data(train_dataloader, 64)

    val_x0 = jax.random.normal(
        jax.random.key(42), val_x1.shape, dtype=val_x1.dtype
    )
    train_x0 = jax.random.normal(
        jax.random.key(42), train_x1.shape, dtype=train_x1.dtype
    )

    fixed_val_batch = {
        'x_1': jax.device_put(val_x1, batch_sharding),
        'x_0': jax.device_put(val_x0, batch_sharding),
        'cond': (
            jax.device_put(val_cond, batch_sharding)
            if val_cond is not None
            else None
        ),
    }
    fixed_train_batch = {
        'x_1': jax.device_put(train_x1, batch_sharding),
        'x_0': jax.device_put(train_x0, batch_sharding),
        'cond': (
            jax.device_put(train_cond, batch_sharding)
            if train_cond is not None
            else None
        ),
    }

    def _eval_fixed_t_step(params, x_1, x_0, cond):
      u = (x_1 - x_0).astype(jnp.float32)
      loss_sum = 0.0
      for t_val in (
          0.05,
          0.17857143,
          0.30714286,
          0.43571429,
          0.56428571,
          0.69285714,
          0.82142857,
          0.95,
      ):
        t_arr = jnp.full((x_1.shape[0],), t_val, dtype=x_1.dtype)
        t_exp = jnp.expand_dims(t_arr, range(1, len(x_1.shape)))
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        pred_v: Any = utils.guided_apply(
            model,
            {'params': params},
            x_t,
            t_arr,
            cond=cond,
            guide=False,
        )
        pred_v = pred_v.astype(jnp.float32)
        loss_sum = loss_sum + jnp.mean((pred_v - u) ** 2)
      return loss_sum / 8.0

    def _solve_posthoc_step(params1, params2, x_1, x_0, cond):
      u = (x_1 - x_0).astype(jnp.float32)
      dot_prod = 0.0
      norm_sq = 0.0
      r2_sq = 0.0
      for t_val in (
          0.05,
          0.17857143,
          0.30714286,
          0.43571429,
          0.56428571,
          0.69285714,
          0.82142857,
          0.95,
      ):
        t_arr = jnp.full((x_1.shape[0],), t_val, dtype=x_1.dtype)
        t_exp = jnp.expand_dims(t_arr, range(1, len(x_1.shape)))
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        v1: Any = utils.guided_apply(
            model,
            {'params': params1},
            x_t,
            t_arr,
            cond=cond,
            guide=False,
        )
        v1 = v1.astype(jnp.float32)
        v2: Any = utils.guided_apply(
            model,
            {'params': params2},
            x_t,
            t_arr,
            cond=cond,
            guide=False,
        )
        v2 = v2.astype(jnp.float32)
        diff = v1 - v2
        r2 = u - v2
        dot_prod = dot_prod + jnp.sum(r2 * diff)
        norm_sq = norm_sq + jnp.sum(diff**2)
        r2_sq = r2_sq + jnp.sum(r2**2)
      c_star = jnp.clip(dot_prod / (norm_sq + 1e-8), 0.0, 1.0)
      n_total = float(u.size * 8)
      opt_loss = (
          r2_sq - 2.0 * c_star * dot_prod + (c_star**2) * norm_sq
      ) / n_total
      return c_star, opt_loss

    cond_shd = batch_sharding if val_cond is not None else None
    p_eval_fixed_t = jax.jit(
        _eval_fixed_t_step,
        in_shardings=(
            replicate_sharding,
            batch_sharding,
            batch_sharding,
            cond_shd,
        ),
        out_shardings=replicate_sharding,
    )
    p_solve_posthoc = jax.jit(
        _solve_posthoc_step,
        in_shardings=(
            replicate_sharding,
            replicate_sharding,
            batch_sharding,
            batch_sharding,
            cond_shd,
        ),
        out_shardings=(replicate_sharding, replicate_sharding),
    )

  autoguidance_self_steps_extra = [
      int(x.strip())
      for x in str(config.get('autoguidance_self_steps_extra', '')).split(',')
      if x.strip()
  ]
  ag_self_extra_params = {}
  ag_self_extra_managers = {}
  for s_step in autoguidance_self_steps_extra:
    mgr = ocp.CheckpointManager(
        directory=ckpt_dir / f'ag_self_{s_step}',
        checkpointers={
            'train_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, cleanup_tmp_directories=True
        ),
    )
    ag_self_extra_managers[s_step] = mgr
    if mgr.latest_step() is not None:
      ag_self_extra_params[s_step] = jax.device_put(
          _restore_params(
              str(ckpt_dir / f'ag_self_{s_step}'),
              -1,
              state.ema_params,
              use_ema=True,
          ),
          replicate_sharding,
      )
      logging.info(
          'Restored extra self-autoguidance snapshot for step %d (saved at'
          ' %d).',
          s_step,
          mgr.latest_step(),
      )

  autoguidance_weights_extra = [
      float(x.strip())
      for x in str(config.get('autoguidance_weights_extra', '')).split(',')
      if x.strip()
  ]

  def _get_dynamic_batch(is_regular_step, batch, config):
    # Dynamic batching: every dynamic_batch_freq steps, use full resolution
    # with smaller batch size to avoid OOM. Other steps use reduced resolution
    # with larger batch size for faster training.
    if config.dynamic_batch_freq > 0:
      if is_regular_step:
        global_batch_size = config.batch_size * config.num_devices
        cur_batch = jax.tree.map(
            lambda x: x[:global_batch_size] if x is not None else None, batch
        )
        cur_n_points = config.n_points
      else:
        cur_batch = batch
        cur_n_points = config.small_n_points
    else:
      cur_batch = batch
      cur_n_points = config.n_points
    return cur_batch, cur_n_points

  # Optional EMA decay switch (off by default, see `effective_polyak_decay`).
  polyak_switch_step = int(config.get('polyak_switch_step', 0))
  polyak_decay_after = float(config.get('polyak_decay_after', 0.0))
  if polyak_switch_step > 0 and config.get('combined_batch_mode', False):
    raise ValueError(
        'polyak_switch_step is not supported with combined_batch_mode'
    )

  def _polyak_for(step):
    return effective_polyak_decay(
        step, config.polyak_decay, polyak_switch_step, polyak_decay_after
    )

  def train_step(state, batch, is_regular_step, scale=None):
    """Performs a single training step."""
    update_rng = jax.random.fold_in(train_rng, state.step)
    # Optional LR annealing (off by default): scales the optimizer update by a
    # factor going from 1 to `lr_decay_final` over `lr_decay_steps` train steps
    # starting at `lr_decay_begin` (train steps restart at 0 on warm-start).
    update_scale = None
    if int(config.get('lr_decay_steps', 0)) > 0:
      update_scale = schedules.anneal_factor(
          state.step,
          begin=int(config.get('lr_decay_begin', 0)),
          steps=int(config.lr_decay_steps),
          final=float(config.get('lr_decay_final', 0.0)),
          shape=str(config.get('lr_decay_shape', 'linear')),
      )
    if scale is not None:  # explicit LR scale (cooldown branches)
      update_scale = scale

    def _exec_pass_fn(
        cur_state,
        sub_batch,
        n_pts,
        pass_rng,
        update_scale=None,
        ext_grad=None,
        ext_w=0.0,
        grad_only=False,
    ):
      sub_key = pass_rng if config.get('random_subsample_rng', False) else None
      coord, feat, cond = utils.prep_data(
          sub_batch,
          config.coord_scale,
          config.feat_scale,
          n_pts // config.n_combine_samples,
          n_combine_samples=config.n_combine_samples,
          use_feat=config.use_feat,
          cond_mode=config.cond_mode,
          dst_index_cond=config.get('dst_index_cond', False),
          add_dummy_cond=config.get('add_dummy_cond', False),
          subsample_key=sub_key,
      )
      if config.get('use_bf16', False):
        coord = coord.astype(jnp.bfloat16)
        if feat is not None:
          feat = feat.astype(jnp.bfloat16)
      class_labels = sub_batch.get('_dataset_index', None)
      if config.train_set == 'pos_neg' and class_labels is not None:
        class_labels = jnp.where(class_labels < 2, 0, 1)
      return utils.update_state(
          model=model,
          state=cur_state,
          optimizer=optimizer,
          coord=coord,
          feat=feat,
          rng=pass_rng,
          schedule=config.schedule,
          polyak_decay=_polyak_for(
              cur_state.step
          ),  # pyrefly: ignore[bad-argument-type]
          cond=cond,
          point_cond=config.point_cond,
          do_ott=config.do_ott,
          reorder_type=config.reorder_type,
          class_labels=class_labels,
          use_class_noise=config.get('use_class_noise', False),
          reorder_noise_strength=config.reorder_noise_strength,
          point_cond_sample_threshold=config.point_cond_sample_threshold,
          feat_cond_dropout_threshold=config.feat_cond_dropout_threshold,
          lambda_cfm=config.get('lambda_cfm', 0.0),
          lambda_cond=config.get('lambda_cond', 0.0),
          use_mst=config.get('use_mst', False),
          k=config.get('k', 1),
          external_grad=ext_grad,
          external_grad_weight=ext_w,
          grad_only=grad_only,
          update_scale=update_scale,
      )

    def _exec_pass(
        sub_batch, n_pts, pass_rng, ext_grad=None, ext_w=0.0, grad_only=False
    ):
      return _exec_pass_fn(
          state,
          sub_batch,
          n_pts,
          pass_rng,
          update_scale=update_scale,
          ext_grad=ext_grad,
          ext_w=ext_w,
          grad_only=grad_only,
      )

    cur_batch, cur_n_points = _get_dynamic_batch(is_regular_step, batch, config)
    return _exec_pass(cur_batch, cur_n_points, update_rng)

  p_train_step = jax.jit(
      train_step,
      in_shardings=(
          replicate_sharding,  # state
          batch_sharding,  # data
      ),
      out_shardings=(
          replicate_sharding,  # state
          replicate_sharding,  # aux
      ),
      static_argnames=('is_regular_step',),
  )
  # Same step with an explicit LR scale (used by `cooldown_dynbatch` branches).
  p_train_step_scaled = jax.jit(
      train_step,
      in_shardings=(
          replicate_sharding,  # state
          batch_sharding,  # data
          replicate_sharding,  # scale
      ),
      out_shardings=(
          replicate_sharding,  # state
          replicate_sharding,  # aux
      ),
      static_argnames=('is_regular_step',),
  )

  def train_step_combined(state, batch, full_batch):
    """Dual-resolution combined batch step with host-sliced full_batch (no SPMD

    collective).
    """
    update_rng = jax.random.fold_in(train_rng, state.step)
    update_scale = None
    if int(config.get('lr_decay_steps', 0)) > 0:
      update_scale = schedules.anneal_factor(
          state.step,
          begin=int(config.get('lr_decay_begin', 0)),
          steps=int(config.lr_decay_steps),
          final=float(config.get('lr_decay_final', 0.0)),
          shape=str(config.get('lr_decay_shape', 'linear')),
      )

    def _exec_pass_c(
        sub_batch, n_pts, pass_rng, ext_grad=None, ext_w=0.0, grad_only=False
    ):
      sub_key = pass_rng if config.get('random_subsample_rng', False) else None
      coord, feat, cond = utils.prep_data(
          sub_batch,
          config.coord_scale,
          config.feat_scale,
          n_pts // config.n_combine_samples,
          n_combine_samples=config.n_combine_samples,
          use_feat=config.use_feat,
          cond_mode=config.cond_mode,
          dst_index_cond=config.get('dst_index_cond', False),
          add_dummy_cond=config.get('add_dummy_cond', False),
          subsample_key=sub_key,
      )
      if config.get('use_bf16', False):
        coord = coord.astype(jnp.bfloat16)
        if feat is not None:
          feat = feat.astype(jnp.bfloat16)
      class_labels = sub_batch.get('_dataset_index', None)
      if config.train_set == 'pos_neg' and class_labels is not None:
        class_labels = jnp.where(class_labels < 2, 0, 1)
      return utils.update_state(
          model=model,
          state=state,
          optimizer=optimizer,
          coord=coord,
          feat=feat,
          rng=pass_rng,
          schedule=config.schedule,
          polyak_decay=config.polyak_decay,
          cond=cond,
          point_cond=config.point_cond,
          do_ott=config.do_ott,
          reorder_type=config.reorder_type,
          class_labels=class_labels,
          use_class_noise=config.get('use_class_noise', False),
          reorder_noise_strength=config.reorder_noise_strength,
          point_cond_sample_threshold=config.point_cond_sample_threshold,
          feat_cond_dropout_threshold=config.feat_cond_dropout_threshold,
          lambda_cfm=config.get('lambda_cfm', 0.0),
          lambda_cond=config.get('lambda_cond', 0.0),
          use_mst=config.get('use_mst', False),
          k=config.get('k', 1),
          external_grad=ext_grad,
          external_grad_weight=ext_w,
          grad_only=grad_only,
          update_scale=update_scale,
      )

    g_large, _ = _exec_pass_c(
        full_batch,
        config.n_points,
        jax.random.fold_in(update_rng, 999),
        grad_only=True,
    )
    k_small = max(1, int(config.get('combined_small_passes', 1)))
    alpha_large = float(config.get('combined_large_grad_weight', 0.5))
    w_single_small = (1.0 - alpha_large) / float(k_small)
    w_ext = 1.0 - w_single_small
    g_ext_accum = jax.tree_util.tree_map(lambda g: alpha_large * g, g_large)
    for m in range(1, k_small):

      def _roll_pts(x, shift_pts=m * int(config.small_n_points)):
        if x is None or x.ndim < 2 or x.shape[1] != config.n_points:
          return x
        return jnp.roll(x, shift=shift_pts, axis=1)

      rolled_batch = jax.tree.map(_roll_pts, batch)
      g_m, _ = _exec_pass_c(
          rolled_batch,
          config.small_n_points,
          jax.random.fold_in(update_rng, m),
          grad_only=True,
      )
      g_ext_accum = jax.tree_util.tree_map(
          lambda acc, gm: acc + w_single_small * gm, g_ext_accum, g_m
      )
    g_ext_norm = jax.tree_util.tree_map(
        lambda acc: acc / max(w_ext, 1e-8), g_ext_accum
    )
    return _exec_pass_c(
        batch,
        config.small_n_points,
        update_rng,
        ext_grad=g_ext_norm,
        ext_w=w_ext,
        grad_only=False,
    )

  p_train_step_combined = jax.jit(
      train_step_combined,
      in_shardings=(
          replicate_sharding,  # state
          batch_sharding,  # batch
          batch_sharding,  # full_batch
      ),
      out_shardings=(
          replicate_sharding,  # state
          replicate_sharding,  # aux
      ),
  )

  def train_step_cooldown(state, batch, scale):
    """Linear LR cooldown step."""
    update_rng = jax.random.fold_in(train_rng, state.step)
    coord, feat, cond = utils.prep_data(
        batch,
        config.coord_scale,
        config.feat_scale,
        config.n_points // config.n_combine_samples,
        n_combine_samples=config.n_combine_samples,
        use_feat=config.use_feat,
        cond_mode=config.cond_mode,
        dst_index_cond=config.get('dst_index_cond', False),
        add_dummy_cond=config.get('add_dummy_cond', False),
        subsample_key=None,
    )
    if config.get('use_bf16', False):
      coord = coord.astype(jnp.bfloat16)
      if feat is not None:
        feat = feat.astype(jnp.bfloat16)
    class_labels = batch.get('_dataset_index', None)
    if config.train_set == 'pos_neg' and class_labels is not None:
      class_labels = jnp.where(class_labels < 2, 0, 1)
    return utils.update_state(
        model=model,
        state=state,
        optimizer=optimizer,
        coord=coord,
        feat=feat,
        rng=update_rng,
        schedule=config.schedule,
        polyak_decay=config.polyak_decay,
        cond=cond,
        point_cond=config.point_cond,
        do_ott=config.do_ott,
        reorder_type=config.reorder_type,
        class_labels=class_labels,
        use_class_noise=config.get('use_class_noise', False),
        reorder_noise_strength=config.reorder_noise_strength,
        point_cond_sample_threshold=config.point_cond_sample_threshold,
        feat_cond_dropout_threshold=config.feat_cond_dropout_threshold,
        lambda_cfm=config.get('lambda_cfm', 0.0),
        lambda_cond=config.get('lambda_cond', 0.0),
        use_mst=config.get('use_mst', False),
        k=config.get('k', 1),
        grad_only=False,
        update_scale=scale,
    )

  p_train_step_cooldown = jax.jit(
      train_step_cooldown,
      in_shardings=(
          replicate_sharding,  # state
          batch_sharding,  # batch
          replicate_sharding,  # scale
      ),
      out_shardings=(
          replicate_sharding,  # state
          replicate_sharding,  # aux
      ),
  )

  def train_step_cooldown_combined(state, batch, full_batch, scale):
    """Linear LR cooldown step for combined batch mode."""
    update_rng = jax.random.fold_in(train_rng, state.step)

    def _exec_pass_c(
        sub_batch, n_pts, pass_rng, ext_grad=None, ext_w=0.0, grad_only=False
    ):
      sub_key = pass_rng if config.get('random_subsample_rng', False) else None
      coord, feat, cond = utils.prep_data(
          sub_batch,
          config.coord_scale,
          config.feat_scale,
          n_pts // config.n_combine_samples,
          n_combine_samples=config.n_combine_samples,
          use_feat=config.use_feat,
          cond_mode=config.cond_mode,
          dst_index_cond=config.get('dst_index_cond', False),
          add_dummy_cond=config.get('add_dummy_cond', False),
          subsample_key=sub_key,
      )
      if config.get('use_bf16', False):
        coord = coord.astype(jnp.bfloat16)
        if feat is not None:
          feat = feat.astype(jnp.bfloat16)
      class_labels = sub_batch.get('_dataset_index', None)
      if config.train_set == 'pos_neg' and class_labels is not None:
        class_labels = jnp.where(class_labels < 2, 0, 1)
      return utils.update_state(
          model=model,
          state=state,
          optimizer=optimizer,
          coord=coord,
          feat=feat,
          rng=pass_rng,
          schedule=config.schedule,
          polyak_decay=config.polyak_decay,
          cond=cond,
          point_cond=config.point_cond,
          do_ott=config.do_ott,
          reorder_type=config.reorder_type,
          class_labels=class_labels,
          use_class_noise=config.get('use_class_noise', False),
          reorder_noise_strength=config.reorder_noise_strength,
          point_cond_sample_threshold=config.point_cond_sample_threshold,
          feat_cond_dropout_threshold=config.feat_cond_dropout_threshold,
          lambda_cfm=config.get('lambda_cfm', 0.0),
          lambda_cond=config.get('lambda_cond', 0.0),
          use_mst=config.get('use_mst', False),
          k=config.get('k', 1),
          external_grad=ext_grad,
          external_grad_weight=ext_w,
          grad_only=grad_only,
          update_scale=scale,
      )

    g_large, _ = _exec_pass_c(
        full_batch,
        config.n_points,
        jax.random.fold_in(update_rng, 999),
        grad_only=True,
    )
    k_small = max(1, int(config.get('combined_small_passes', 1)))
    alpha_large = float(config.get('combined_large_grad_weight', 0.5))
    w_single_small = (1.0 - alpha_large) / float(k_small)
    w_ext = 1.0 - w_single_small
    g_ext_accum = jax.tree_util.tree_map(lambda g: alpha_large * g, g_large)
    for m in range(1, k_small):

      def _roll_pts(x, shift_pts=m * int(config.small_n_points)):
        if x is None or x.ndim < 2 or x.shape[1] != config.n_points:
          return x
        return jnp.roll(x, shift=shift_pts, axis=1)

      rolled_batch = jax.tree.map(_roll_pts, batch)
      g_m, _ = _exec_pass_c(
          rolled_batch,
          config.small_n_points,
          jax.random.fold_in(update_rng, m),
          grad_only=True,
      )
      g_ext_accum = jax.tree_util.tree_map(
          lambda acc, gm: acc + w_single_small * gm, g_ext_accum, g_m
      )
    g_ext_norm = jax.tree_util.tree_map(
        lambda acc: acc / max(w_ext, 1e-8), g_ext_accum
    )
    return _exec_pass_c(
        batch,
        config.small_n_points,
        update_rng,
        ext_grad=g_ext_norm,
        ext_w=w_ext,
        grad_only=False,
    )

  p_train_step_cooldown_combined = jax.jit(
      train_step_cooldown_combined,
      in_shardings=(
          replicate_sharding,  # state
          batch_sharding,  # batch
          batch_sharding,  # full_batch
          replicate_sharding,  # scale
      ),
      out_shardings=(
          replicate_sharding,  # state
          replicate_sharding,  # aux
      ),
  )

  def val_step(state, batch, val_rng):
    """Computes the validation loss."""

    coord, feat, cond = utils.prep_data(
        batch,
        config.coord_scale,
        config.feat_scale,
        config.n_points // config.n_combine_samples,
        n_combine_samples=config.n_combine_samples,
        use_feat=config.use_feat,
        cond_mode=config.cond_mode,
        dst_index_cond=config.get('dst_index_cond', False),
        add_dummy_cond=config.get('add_dummy_cond', False),
    )

    if config.get('use_bf16', False):
      coord = coord.astype(jnp.bfloat16)
      if feat is not None:
        feat = feat.astype(jnp.bfloat16)

    variables = {'params': state.ema_params}
    if state.batch_stats:
      variables['batch_stats'] = state.batch_stats
    return utils.compute_loss(
        model,
        variables,
        val_rng,
        coord=coord,
        feat=feat,
        schedule=config.schedule,
        is_train=False,
        cond=cond,
        point_cond=config.point_cond,
        do_ott=config.do_ott,
        reorder_type=config.reorder_type,
        reorder_noise_strength=config.reorder_noise_strength,
        point_cond_sample_threshold=config.point_cond_sample_threshold,
        feat_cond_dropout_threshold=config.feat_cond_dropout_threshold,
        lambda_cfm=config.get('lambda_cfm', 0.0),
        lambda_cond=config.get('lambda_cond', 0.0),
        use_mst=config.get('use_mst', False),
    )[0]

  p_val_step = jax.jit(
      val_step,
      in_shardings=(
          replicate_sharding,  # state
          batch_sharding,  # data
          replicate_sharding,  # rng
      ),
      out_shardings=replicate_sharding,  # loss
  )

  def generate(
      state,
      rng,
      cond,
      noise,
      point_cond_mask,
      guidance_scale,
      guide,
      class_labels=None,
      ag_params=None,
  ):
    """Generates samples from the model."""
    sample_shape = (init_coord.shape[1], init_coord.shape[-1] + feat_shape)
    # Eval-time sampler options; the defaults add no kwargs (unchanged path).
    if config.optimizer == 'schedule_free_adamw':

      def find_sf_state(s, p):
        if isinstance(s, tuple):
          for x in s:
            res = find_sf_state(x, p)
            if res is not None:
              return res
        elif hasattr(s, 'inner_opt_state'):
          return find_sf_state(s.inner_opt_state, p)
        elif hasattr(s, 'z'):
          return optax.contrib.schedule_free_eval_params(s, p)
        return None

      sf_eval_params = find_sf_state(state.opt_state, state.params)
      if sf_eval_params is not None:
        state = state.replace(ema_params=sf_eval_params)
    elif config.get('sample_posthoc_ema', False) and posthoc_emas is not None:
      c_val = float(config.get('posthoc_c_star', 0.5))
      state = state.replace(
          ema_params=utils.reconstruct_posthoc_ema(posthoc_emas, c_val)
      )
    elif not config.get('sample_use_ema', True):
      state = state.replace(ema_params=state.params)  # raw, non-Polyak params
    extra_kwargs = {}
    if config.get('sample_solver', 'midpoint') != 'midpoint':
      extra_kwargs['solver'] = config.sample_solver
    if ag_params is not None:
      extra_kwargs['autoguidance_params'] = ag_params
      extra_kwargs['autoguidance_weight'] = float(
          config.get('autoguidance_weight', 1.0)
      )
      ag_t_min = float(config.get('autoguidance_t_min', 0.0))
      ag_t_max = float(config.get('autoguidance_t_max', 1.0))
      if ag_t_min > 0.0 or ag_t_max < 1.0:
        extra_kwargs['autoguidance_t_min'] = ag_t_min
        extra_kwargs['autoguidance_t_max'] = ag_t_max
      if config.get('ag_decouple_scale', False):
        extra_kwargs['ag_decouple_scale'] = True
    if float(config.get('sample_churn', 0.0)) > 0.0:
      extra_kwargs['churn'] = float(config.sample_churn)
      if 'sample_churn_tmin' in config:
        extra_kwargs['churn_tmin'] = float(config.sample_churn_tmin)
      if 'sample_churn_tmax' in config:
        extra_kwargs['churn_tmax'] = float(config.sample_churn_tmax)
    return utils.generate_samples(
        model,
        state,
        config.num_devices,
        sample_shape,
        rng,
        config.sample_steps,
        config.sample_schedule,
        cond=cond,
        noise=noise,
        point_cond_mask=point_cond_mask,
        guidance_scale=guidance_scale,
        guide=guide,
        use_class_noise=config.get('use_class_noise', False),
        class_labels=class_labels,
        **extra_kwargs,
    )

  p_generate = jax.jit(
      generate,
      in_shardings=(
          replicate_sharding,  # state
          replicate_sharding,  # rng
          batch_sharding,  # cond
          batch_sharding,  # noise
          batch_sharding,  # point_cond_mask
          batch_sharding,  # guidance_scale
          batch_sharding,  # class_labels
      ),
      out_shardings=batch_sharding,  # samples
      static_argnames=('guide',),
  )
  p_generate_ag = jax.jit(
      generate,
      in_shardings=(
          replicate_sharding,  # state
          replicate_sharding,  # rng
          batch_sharding,  # cond
          batch_sharding,  # noise
          batch_sharding,  # point_cond_mask
          batch_sharding,  # guidance_scale
          batch_sharding,  # class_labels
          replicate_sharding,  # ag_params
      ),
      out_shardings=batch_sharding,  # samples
      static_argnames=('guide',),
  )

  def generate_chunk(state, rng, cond, point_cond_mask, class_labels):
    """Unconditional generation for the metrics (optionally autoguided)."""
    if autoguidance_params is None:
      return p_generate(
          state, rng, cond, None, point_cond_mask, None, False, class_labels
      )
    return p_generate_ag(
        state,
        rng,
        cond,
        None,
        point_cond_mask,
        None,
        False,
        class_labels,
        autoguidance_params,
    )

  # Optional extra evaluations (off by default): `eval_extra_draws` more
  # independent 512-cloud draws of the headline sampler, and sampler variants
  # (`eval_variants`, see `parse_eval_variants`) scored on the same draws.
  n_extra_draws = int(config.get('eval_extra_draws', 0))
  # Also log diversity stats (std ratio, coverage, ...; see
  # `utils.diversity_metrics`) for the headline and extra draws.
  eval_diversity = bool(config.get('eval_diversity', False))
  eval_variants = parse_eval_variants(
      str(config.get('eval_variants', '')),
      has_autoguidance=autoguidance_params is not None or ag_self_step > 0,
  )
  base_sampler = {
      'steps': int(config.sample_steps),
      'sched': str(config.sample_schedule),
      'solver': str(config.get('sample_solver', 'midpoint')),
      'w': float(config.get('autoguidance_weight', 1.0)),
      'tmin': float(config.get('autoguidance_t_min', 0.0)),
      'tmax': float(config.get('autoguidance_t_max', 1.0)),
      'temp': 1.0,
      'churn': float(config.get('sample_churn', 0.0)),
      'ctmin': float(config.get('sample_churn_tmin', 0.02)),
      'ctmax': float(config.get('sample_churn_tmax', 0.95)),
      'ds': bool(config.get('ag_decouple_scale', False)),
  }

  def make_variant_generate(sampler):
    """Jitted unconditional generator for one sampler variant."""
    # Whether the variant needs a guide; the guide itself is looked up at call
    # time because a self-autoguidance snapshot appears only mid-run.
    wants_ag = sampler['w'] != 1.0

    def gen(state, rng, cond, point_cond_mask, class_labels, ag_params):
      if config.get('sample_posthoc_ema', False) and posthoc_emas is not None:
        c_val = float(config.get('posthoc_c_star', 0.5))
        state = state.replace(
            ema_params=utils.reconstruct_posthoc_ema(posthoc_emas, c_val)
        )
      elif not config.get('sample_use_ema', True):
        state = state.replace(ema_params=state.params)
      return utils.generate_samples(
          model,
          state,
          config.num_devices,
          (init_coord.shape[1], init_coord.shape[-1] + feat_shape),
          rng,
          sampler['steps'],
          sampler['sched'],
          cond=cond,
          point_cond_mask=point_cond_mask,
          solver=sampler['solver'],
          use_class_noise=config.get('use_class_noise', False),
          class_labels=class_labels,
          autoguidance_params=ag_params,
          autoguidance_weight=sampler['w'],
          autoguidance_t_min=sampler['tmin'],
          autoguidance_t_max=sampler['tmax'],
          noise_temp=sampler['temp'],
          churn=sampler['churn'],
          churn_tmin=sampler['ctmin'],
          churn_tmax=sampler['ctmax'],
          ag_decouple_scale=sampler.get('ds', False),
      )

    p_gen = jax.jit(
        gen,
        in_shardings=(
            replicate_sharding,  # state
            replicate_sharding,  # rng
            batch_sharding,  # cond
            batch_sharding,  # point_cond_mask
            batch_sharding,  # class_labels
            replicate_sharding,  # ag_params
        ),
        out_shardings=batch_sharding,
    )
    return wants_ag, lambda state, rng, cond, pcm, cl: p_gen(
        state, rng, cond, pcm, cl, autoguidance_params if wants_ag else None
    )

  variant_generators = []
  variant_wants_ag = {}
  for v_name, v_overrides in eval_variants:
    v_wants_ag, v_gen = make_variant_generate(dict(base_sampler, **v_overrides))
    variant_generators.append((v_name, v_gen))
    variant_wants_ag[v_name] = v_wants_ag

  extra_weight_generators = {}
  for w_val in autoguidance_weights_extra:
    _, w_gen = make_variant_generate(dict(base_sampler, w=w_val))
    extra_weight_generators[w_val] = w_gen

  def score_draw(
      gen_fn,
      state,
      draw_rng,
      cond,
      point_cond_mask,
      class_labels,
      n_chunks,
      n_gen,
  ):
    """Generates one draw of `n_gen` clouds.

    Returns:
      (s_mmd_val_mst, s_mmd_val, diversity dict, s_mmd_train_mst).
    """
    x = jnp.concatenate(
        [
            gen_fn(
                state,
                jax.random.fold_in(draw_rng, i),
                cond,
                point_cond_mask,
                class_labels[
                    i * config.num_devices : (i + 1) * config.num_devices
                ],
            )
            for i in range(n_chunks)
        ],
        axis=0,
    )[:n_gen]
    x = x[:, :, :3] / config.coord_scale
    should_rotate = (
        config.eval_rotate_samples
        if getattr(config, 'eval_rotate_samples', None) is not None
        else not config.do_rotate
    )
    if should_rotate:
      x = x @ rotations.random_rotation(draw_rng, num=x.shape[0])
    x = multihost_utils.process_allgather(x, tiled=True)
    if eval_diversity:
      m, div = utils.compute_metrics_and_diversity(x, config)
    else:
      m, div = utils.compute_metrics(x, config), {}
    return (
        float(jax.device_get(m[1])),
        float(jax.device_get(m[5])),
        div,
        float(jax.device_get(m[0])),
    )

  def extra_evals(
      state,
      step,
      headline_rng,
      headline,
      cond,
      point_cond_mask,
      class_labels,
      n_chunks,
      n_gen,
  ):
    """Scores extra draws / sampler variants; returns scalars to log."""
    draw_rngs = [headline_rng] + [
        jax.random.fold_in(
            jax.random.fold_in(
                jax.random.key(config.generation_seed), 1_000_003 + k
            ),
            step,
        )
        for k in range(1, n_extra_draws + 1)
    ]
    out = {}
    settings = [('', None)] + [
        (name, gen_fn)
        for name, gen_fn in variant_generators
        if autoguidance_params is not None or not variant_wants_ag[name]
    ]
    for name, gen_fn in settings:
      prefix = f'ev_{name}/' if name else ''
      vals_mst, vals, divs, vals_train = [], [], [], []
      for k, draw_rng in enumerate(draw_rngs):
        if gen_fn is None and k == 0:
          mst, sub, div, train_mst = headline  # headline draw, already scored
        else:
          mst, sub, div, train_mst = score_draw(
              gen_fn if gen_fn is not None else generate_chunk,
              state,
              draw_rng,
              cond,
              point_cond_mask,
              class_labels,
              n_chunks,
              n_gen,
          )
          both_mst = 0.5 * (mst + train_mst)
          out[f'{prefix}s_mmd_val_mst_d{k}'] = mst
          out[f'{prefix}s_mmd_val_d{k}'] = sub
          out[f'{prefix}s_mmd_train_mst_d{k}'] = train_mst
          out[f'{prefix}s_mmd_both_mst_d{k}'] = both_mst
          for key, value in div.items():
            out[f'{prefix}div_{key}_d{k}'] = value
        vals_mst.append(mst)
        vals.append(sub)
        divs.append(div)
        vals_train.append(train_mst)
      out[f'{prefix}s_mmd_val_mst_mean'] = float(np.mean(vals_mst))
      out[f'{prefix}s_mmd_val_mean'] = float(np.mean(vals))
      out[f'{prefix}s_mmd_train_mst_mean'] = float(np.mean(vals_train))
      out[f'{prefix}s_mmd_both_mst_mean'] = float(
          0.5 * (np.mean(vals_mst) + np.mean(vals_train))
      )
      for key in divs[0] if all(divs) else ():
        out[f'{prefix}div_{key}_mean'] = float(np.mean([d[key] for d in divs]))
    return out

  running_class_1_sum = 0.0
  running_total_count = 0.0
  initial_step = int(jax.device_get(state.step))

  cooldown_start_step = int(config.get('cooldown_start_step', 0))
  cooldown_branches = [
      int(x.strip())
      for x in str(config.get('cooldown_branches', '')).split(',')
      if x.strip()
  ]
  target_main_steps = (
      cooldown_start_step
      if (cooldown_start_step > 0 and cooldown_branches)
      else config.max_steps
  )

  generation_step_count = 0

  if config.get('combined_batch_mode', False):
    n_dev = int(config.num_devices)
    full_bs_per_dev = max(
        1,
        min(
            int(config.large_batch_size),
            int(config.get('combined_full_batch_size', 8)),
        ),
    )

    def _slice_to_full(x):
      if x is None:
        return None
      per_dev = x.shape[0] // n_dev
      return np.concatenate(
          [
              x[i * per_dev : i * per_dev + full_bs_per_dev]
              for i in range(n_dev)
          ],
          axis=0,
      )

  else:
    _slice_to_full = None

  last_checkpoint_time = time.time()

  while jax.device_get(state.step) < target_main_steps:
    batch = next(train_iter)

    batch = {
        'coord': batch['coord'],
        'feat': batch['feat'],
        '_dataset_index': (
            batch['_dataset_index'] if '_dataset_index' in batch else None
        ),
    }
    assert ((-1 < batch['coord']) & (batch['coord'] < 1)).mean() > 0.9

    if batch['_dataset_index'] is not None:
      class_labels = batch['_dataset_index']
      if config.train_set == 'pos_neg':
        class_labels = np.where(class_labels < 2, 0, 1)
      running_class_1_sum += float(np.sum(class_labels))
      running_total_count += int(class_labels.shape[0])

    if config.get('combined_batch_mode', False) and _slice_to_full is not None:
      full_batch_np = jax.tree.map(_slice_to_full, batch)
      batch = jax.tree.map(jnp.array, batch)
      full_batch = jax.tree.map(jnp.array, full_batch_np)
      state, aux = p_train_step_combined(state, batch, full_batch)
      utils.log_bins(log_dict, aux)
      running_train_loss += aux['loss']
      running_train_loss_squared += aux['loss'] ** 2
      running_aux_loss += aux.get('aux_loss', 0.0)
      running_train_step += 1
    elif (
        config.get('lr', 0.0) == 0.0 and config.get('polyak_decay', 0.0) >= 1.0
    ):
      batch = jax.tree.map(jnp.array, batch)
      # pyrefly: ignore[missing-attribute]
      state = state.replace(step=state.step + 1)
      running_train_step += 1
    else:
      batch = jax.tree.map(jnp.array, batch)
      step = int(jax.device_get(state.step))
      is_regular_step = (
          (step % config.dynamic_batch_freq == 0)
          if config.dynamic_batch_freq > 0
          else True
      )
      state, aux = p_train_step(state, batch, is_regular_step)
      utils.log_bins(log_dict, aux)
      running_train_loss += aux['loss']
      running_train_loss_squared += aux['loss'] ** 2
      running_aux_loss += aux.get('aux_loss', 0.0)
      running_train_step += 1

    if p_update_extra_emas is not None:
      extra_emas = p_update_extra_emas(extra_emas, state.params, state.step)
    if p_update_posthoc_emas is not None:
      posthoc_emas = p_update_posthoc_emas(
          posthoc_emas, state.params, state.step
      )
    if p_guide_ema_update is not None and guide_ema is not None:
      # Same arithmetic and step index as the main EMA in utils.update_state.
      guide_ema = p_guide_ema_update(guide_ema, state.params, state.step - 1)

    step = int(jax.device_get(state.step))
    for s_step in autoguidance_self_steps_extra:
      if s_step not in ag_self_extra_params and step >= s_step:
        p_snap = jax.device_put(
            jax.tree_util.tree_map(jnp.copy, state.ema_params),
            replicate_sharding,
        )
        ag_self_extra_params[s_step] = p_snap
        if s_step in ag_self_extra_managers:
          ag_self_extra_managers[s_step].save(
              step, items={'train_state': {'ema_params': p_snap}}
          )
          ag_self_extra_managers[s_step].wait_until_finished()
        logging.info(
            'Took and persisted extra self-autoguidance snapshot at step %d for'
            ' target %d.',
            step,
            s_step,
        )

    step = int(jax.device_get(state.step))
    # One sanity log/ckpt/eval/generation right after a warm start from
    # init_ckpt_path; preemption resumes keep the absolute-step grid only.
    warm_sanity = warm_started and step == initial_step + 10

    if (
        ag_self_manager is not None
        and autoguidance_params is None
        and step >= ag_self_step
    ):
      # Copy so later EMA updates can never alias the guide's buffers.
      autoguidance_params = jax.device_put(
          jax.tree_util.tree_map(
              jnp.copy, guide_ema if guide_ema is not None else state.ema_params
          ),
          replicate_sharding,
      )
      ag_self_manager.save(
          step, items={'train_state': {'ema_params': autoguidance_params}}
      )
      ag_self_manager.wait_until_finished()
      logging.info(
          'Took self-autoguidance snapshot at step %d (%s).',
          step,
          f'guide EMA {guide_ema_decay}'
          if guide_ema is not None
          else 'main EMA',
      )
      guide_ema = None  # no longer tracked or saved after the snapshot

    save_checkpoint_steps = int(config.get('save_checkpoint_steps', 10000))
    save_checkpoint_secs = int(
        config.get('save_checkpoint_secs', 0)
        or (
            FLAGS.save_checkpoint_secs
            if 'save_checkpoint_secs' in flags.FLAGS
            and FLAGS.save_checkpoint_secs
            else 0
        )
    )
    should_save, is_wall_clock = should_save_checkpoint(
        step=step,
        save_checkpoint_steps=save_checkpoint_steps,
        last_checkpoint_time=last_checkpoint_time,
        save_checkpoint_secs=save_checkpoint_secs,
        now=time.time(),
        max_steps=config.max_steps,
        warm_sanity=warm_sanity,
    )

    if (
        (
            config.log_train_every_steps > 0
            and step % config.log_train_every_steps == 0
        )
        or should_save
        or step == config.max_steps
        or step == 10
        or step == 20
        or warm_sanity
    ):
      train_loss = (
          running_train_loss / running_train_step
          if running_train_step > 0
          else 0.0
      )
      train_loss_variance = (
          running_train_loss_squared / running_train_step - train_loss**2
          if running_train_step > 0
          else 0.0
      )
      class_ratio = (
          running_class_1_sum / running_total_count
          if running_total_count > 0
          else 0.5
      )
      if jax.process_index() == 0:
        logging.info(
            'Step: %d, Train Loss: %.5f, Train Loss Variance: %.5f, Class'
            ' Ratio: %.5f, Time: %.2fs',
            step,
            train_loss,
            train_loss_variance,
            class_ratio,
            time.time() - train_start_time,
        )
      writer.write_scalars(
          step * global_batch_size,
          {
              'train_loss': train_loss,
              'grad_step': step,
              'steps_per_sec': (
                  running_train_step / max(1e-4, time.time() - train_start_time)
              ),
          },
      )
      writer.write_scalars(
          step * global_batch_size, {'class_ratio': class_ratio}
      )
      writer.write_scalars(
          step * global_batch_size,
          {'train_loss_variance': train_loss_variance},
      )
      writer.write_scalars(
          step * global_batch_size, utils.log_dict_to_scalars(log_dict)
      )
      writer.write_scalars(
          step * global_batch_size,
          {
              'aux_loss': (
                  running_aux_loss / running_train_step
                  if running_train_step > 0
                  else 0.0
              )
          },
      )

      if should_save:
        last_checkpoint_time = time.time()
        if is_wall_clock and jax.process_index() == 0:
          logging.info(
              'Wall-clock checkpoint save triggered at step %d (%.1fs elapsed).',
              step,
              save_checkpoint_secs,
          )
        if guide_ema_manager is not None and guide_ema is not None:
          # Saved first: every restorable train-state step has a guide EMA.
          guide_ema_manager.save(step, items={'guide_ema': guide_ema})
          guide_ema_manager.wait_until_finished()
        latest_checkpoint_manager.save(
            step,
            items={
                'train_state': state,
                'train_iter': train_iter,
            },
        )
        latest_checkpoint_manager.wait_until_finished()
        if extra_ema_manager is not None:
          extra_ema_manager.save(step, items={'extra_emas': extra_emas})
          extra_ema_manager.wait_until_finished()
        if posthoc_ema_manager is not None:
          posthoc_ema_manager.save(step, items={'posthoc_emas': posthoc_emas})
          posthoc_ema_manager.wait_until_finished()

      running_train_loss = 0.0
      running_train_loss_squared = 0.0
      running_aux_loss = 0.0
      running_train_step = 0
      log_dict = {
          'sum': collections.defaultdict(float),
          'count': collections.defaultdict(int),
      }
      train_start_time = time.time()

    if (
        (config.eval_every_steps > 0 and step % config.eval_every_steps == 0)
        or step == config.max_steps
        or step == 10
        or step == 20
        or warm_sanity
    ):

      val_start_time = time.time()
      running_val_loss = 0.0
      for i, batch in tqdm.tqdm(enumerate(val_dataloader), mininterval=10):
        # TODO(riegerfr): assert always same order/samples (no race conditions)
        batch = {
            'coord': batch['coord'],
            'feat': batch['feat'],
            '_dataset_index': (
                batch['_dataset_index'] if '_dataset_index' in batch else None
            ),
        }
        assert ((-1 < batch['coord']) & (batch['coord'] < 1)).mean() > 0.9

        batch = jax.tree.map(jnp.array, batch)
        running_val_loss += p_val_step(
            state, batch, jax.random.fold_in(val_rng, i)
        )
        if i + 1 >= config.eval_steps:
          break
      # pylint: disable=undefined-loop-variable
      val_loss = running_val_loss / (i + 1)
      if jax.process_index() == 0:
        logging.info(
            'Step: %d, Val Loss: %.5f, Time: %.2fs',
            step,
            val_loss,
            time.time() - val_start_time,
        )
      writer.write_scalars(step * global_batch_size, {'val_loss': val_loss})
      if (
          fixed_t_eval_active
          and p_eval_fixed_t is not None
          and fixed_val_batch is not None
          and fixed_train_batch is not None
      ):
        v_loss_fixed = p_eval_fixed_t(
            state.ema_params,
            fixed_val_batch['x_1'],
            fixed_val_batch['x_0'],
            fixed_val_batch['cond'],
        )
        t_loss_fixed = p_eval_fixed_t(
            state.ema_params,
            fixed_train_batch['x_1'],
            fixed_train_batch['x_0'],
            fixed_train_batch['cond'],
        )
        fixed_t_scalars = {
            'eval/val_loss_fixed_t': float(v_loss_fixed),
            'eval/train_loss_fixed_t': float(t_loss_fixed),
        }
        if (
            posthoc_ema
            and p_solve_posthoc is not None
            and posthoc_emas is not None
        ):
          c_star, ph_loss = p_solve_posthoc(
              posthoc_emas['ema_gamma1'],
              posthoc_emas['ema_gamma2'],
              fixed_val_batch['x_1'],
              fixed_val_batch['x_0'],
              fixed_val_batch['cond'],
          )
          fixed_t_scalars['eval/posthoc_c_star'] = float(c_star)
          fixed_t_scalars['eval/val_loss_posthoc_fixed_t'] = float(ph_loss)
        writer.write_scalars(step * global_batch_size, fixed_t_scalars)
        logging.info(
            'Step: %d, Fixed-t Val Loss: %.5f, Fixed-t Train Loss: %.5f',
            step,
            float(v_loss_fixed),
            float(t_loss_fixed),
        )
        if 'eval/posthoc_c_star' in fixed_t_scalars:
          logging.info(
              'Step: %d, Post-hoc EMA c*: %.4f, Post-hoc Val Loss: %.5f',
              step,
              fixed_t_scalars['eval/posthoc_c_star'],
              fixed_t_scalars['eval/val_loss_posthoc_fixed_t'],
          )

    if (
        (
            config.generation_every_steps > 0
            and step % config.generation_every_steps == 0
        )
        or step == config.max_steps
        or step == 10
        or step == 20
        or warm_sanity
    ):
      generation_start_time = time.time()
      generation_step_count += 1
      if init_cond is not None:
        cond = jnp.concatenate(
            (jnp.zeros_like(init_cond), jnp.zeros_like(init_cond)), axis=-1
        )[: config.num_devices]
      else:
        cond = None
      if config.point_cond > 0:
        point_cond_mask = jnp.zeros(
            (
                init_coord.shape[0],
                init_coord.shape[1],
            ),
            dtype=bool,
        )[: config.num_devices]
      else:
        point_cond_mask = None

      if config.train_set == 'pos_neg':
        n_samples_to_gen = 1024
        class_labels_all = jnp.concatenate(
            [jnp.zeros(512, dtype=jnp.int32), jnp.ones(512, dtype=jnp.int32)]
        )
        n_gen_chunks = n_samples_to_gen // config.num_devices
      else:
        n_samples_to_gen = config.n_samples
        ratio = (
            running_class_1_sum / running_total_count
            if running_total_count > 0
            else 0.5
        )
        rng_gen, generation_rng = jax.random.split(generation_rng)
        n_gen_chunks = max(1, n_samples_to_gen // config.num_devices)
        class_labels_all = jax.random.bernoulli(
            rng_gen, ratio, (n_gen_chunks * config.num_devices,)
        ).astype(jnp.int32)

      x_gen_full = jnp.concatenate(
          [
              generate_chunk(
                  state,
                  jax.random.fold_in(generation_rng, i),
                  cond,
                  point_cond_mask,
                  class_labels_all[
                      i * config.num_devices : (i + 1) * config.num_devices
                  ],
              )
              for i in range(n_gen_chunks)
          ],
          axis=0,
      )[:n_samples_to_gen]
      x_gen = x_gen_full[:, :, :3] / config.coord_scale
      if config.use_feat:
        x_feat_gen = x_gen_full[:, :, 3:] / config.feat_scale
      else:
        x_feat_gen = None

      should_rotate = (
          config.eval_rotate_samples
          if getattr(config, 'eval_rotate_samples', None) is not None
          else not config.do_rotate
      )
      if should_rotate:
        # training without rotation, rotate here to get same eval statistics
        x_gen = x_gen @ rotations.random_rotation(
            generation_rng, num=x_gen.shape[0]
        )

      # for multihost TPU support
      # TODO(riegerfr): strictly necessary?
      x_gen_gathered = multihost_utils.process_allgather(x_gen, tiled=True)
      if config.use_feat and x_feat_gen is not None:
        x_feat_gen_gathered = multihost_utils.process_allgather(
            x_feat_gen, tiled=True
        )
      else:
        x_feat_gen_gathered = None
      if jax.process_index() == 0:
        for target_dir in [workdir, log_path]:
          target_dir.mkdir(parents=True, exist_ok=True)
          with storage.atomic_file(
              str(target_dir / 'x_gen.html'),
              'w',  # local=False
          ) as f:
            utils.plot_point_clouds(
                x_gen_gathered[: utils.N_PLOT],
                n_combine_samples=config.n_combine_samples,
            ).write_html(f)
      metrics_start_time = time.time()
      mmd_train, mmd_val, fid_train, fid_val = 1000.0, 1000.0, 1000.0, 1000.0
      if jax.process_index() == 0:
        logging.info('x_gen shape: %r', x_gen.shape)
        logging.info('batch["coord"].shape: %r', batch['coord'].shape)
      assert len(x_gen.shape) == 3  # batch, n_points, 3
      multihost_utils.process_allgather(jax.numpy.array(0))
      logging.debug('debug: Completed all-gather barrier.')
      headline_div = {}
      if config.train_set == 'pos_neg':
        # pyrefly: ignore[bad-argument-type]
        config_pos = ml_collections.ConfigDict(config)
        config_pos.train_set = 'train'
        # pyrefly: ignore[bad-argument-type]
        config_neg = ml_collections.ConfigDict(config)
        config_neg.train_set = 'axons_negative'

        class_labels_gathered = jnp.concatenate(
            [class_labels_all] * jax.process_count()
        )
        pos_mask = class_labels_gathered == 0
        neg_mask = class_labels_gathered == 1

        metrics_pos = utils.compute_metrics(
            x_gen_gathered[pos_mask], config_pos
        )
        metrics_neg = utils.compute_metrics(
            x_gen_gathered[neg_mask], config_neg
        )

        s_mmd_train_pos = metrics_pos[0]
        s_mmd_train_neg = metrics_neg[0]

        writer.write_scalars(
            step * global_batch_size,
            {
                's_mmd_train_pos': s_mmd_train_pos,
                's_mmd_train_neg': s_mmd_train_neg,
            },
        )

        # Average for main metrics used in early stopping and logging
        metrics_results = (
            (metrics_pos[0] + metrics_neg[0]) / 2.0,
            (metrics_pos[1] + metrics_neg[1]) / 2.0,
            (metrics_pos[2] + metrics_neg[2]) / 2.0,
            (metrics_pos[3] + metrics_neg[3]) / 2.0,
            (metrics_pos[4] + metrics_neg[4]) / 2.0,
            (metrics_pos[5] + metrics_neg[5]) / 2.0,
        )
      else:
        if eval_diversity:
          metrics_results, headline_div = utils.compute_metrics_and_diversity(
              x_gen_gathered, config, with_reference=True
          )
        else:
          metrics_results = utils.compute_metrics(x_gen_gathered, config)

      (
          s_mmd_train,
          s_mmd_val,
          s_fid_train,
          s_fid_val,
          s_mmd_train_sub,
          s_mmd_val_sub,
      ) = metrics_results

      logging.info(
          'computed metrics, s_mmd_train: %r, s_mmd_val_mst: %r',
          s_mmd_train,
          s_mmd_val,
      )

      if jax.device_get(s_mmd_train) < jax.device_get(state.min_s_mmd_train):
        state = state.replace(min_s_mmd_train=jax.device_get(s_mmd_train))
        best_x_gen = x_gen
        if jax.process_index() == 0:
          with storage.atomic_file(
              str(workdir / 'best_s_mmd_train.npz'),
              'wb',
              #   local=False,
          ) as f:
            save_dict = {'coord': np.asarray(x_gen_gathered)}
            if config.use_feat and x_feat_gen_gathered is not None:
              save_dict['feat'] = np.asarray(x_feat_gen_gathered)
            np.savez_compressed(f, **save_dict)

          with storage.atomic_file(
              str(workdir / 'best_s_mmd_train.html'),
              'w',
          ) as f:
            utils.plot_point_clouds(
                x_gen_gathered[: utils.N_PLOT],
                n_combine_samples=config.n_combine_samples,
            ).write_html(f)

        if config.point_cond > 0:
          utils.log_point_cond(
              state,
              cond,  # pyrefly: ignore[bad-argument-type]
              init_coord,
              init_feat,
              workdir,
              config,
              generation_rng,
              p_generate,
          )

        if init_cond is not None:
          utils.log_conditioned(
              generation_rng,
              state,
              init_coord,
              init_feat,
              init_cond,
              workdir,
              config,
              p_generate,
          )
        best_checkpoint_manager.save(
            step,
            items={
                'train_state': state,
            },
        )
        best_checkpoint_manager.wait_until_finished()

      extra_scalars = {}
      if (n_extra_draws > 0 or variant_generators) and (
          config.train_set != 'pos_neg'
      ):
        extra_start_time = time.time()
        extra_scalars = extra_evals(
            state,
            step,
            generation_rng,
            (
                float(jax.device_get(s_mmd_val)),
                float(jax.device_get(s_mmd_val_sub)),
                {
                    k: v
                    for k, v in headline_div.items()
                    if not k.startswith('ref_')
                },
                float(jax.device_get(s_mmd_train)),
            ),
            cond,
            point_cond_mask,
            class_labels_all,
            n_gen_chunks,
            n_samples_to_gen,
        )
        extra_scalars['extra_eval_time'] = time.time() - extra_start_time
        logging.info('Step: %d, extra evals: %r', step, extra_scalars)

      is_every_5th_gen = (
          generation_step_count % 5 == 0 or step == config.max_steps
      )
      if is_every_5th_gen and config.train_set != 'pos_neg':
        for d in extra_ema_decays:
          k_ema = f'ema_{d}'
          if k_ema in extra_emas:
            s_ema = state.replace(ema_params=extra_emas[k_ema])
            mst, sub, div, train_mst = score_draw(
                generate_chunk,
                s_ema,
                generation_rng,
                cond,
                point_cond_mask,
                class_labels_all,
                n_gen_chunks,
                n_samples_to_gen,
            )
            extra_scalars[f'{k_ema}/s_mmd_val_mst'] = mst
            extra_scalars[f'{k_ema}/s_mmd_val'] = sub
            extra_scalars[f'{k_ema}/s_mmd_train_mst'] = train_mst
            extra_scalars[f'{k_ema}/s_mmd_both_mst'] = 0.5 * (mst + train_mst)

        for s_step in autoguidance_self_steps_extra:
          if s_step in ag_self_extra_params:
            tag = f'self_ag_{s_step // 1000}k'
            ag_p = ag_self_extra_params[s_step]
            gen_fn = lambda s, r, c, pcm, cl: p_generate_ag(
                s, r, c, None, pcm, None, False, cl, ag_p
            )
            mst, sub, div, train_mst = score_draw(
                gen_fn,
                state,
                generation_rng,
                cond,
                point_cond_mask,
                class_labels_all,
                n_gen_chunks,
                n_samples_to_gen,
            )
            extra_scalars[f'{tag}/s_mmd_val_mst'] = mst
            extra_scalars[f'{tag}/s_mmd_val'] = sub
            extra_scalars[f'{tag}/s_mmd_train_mst'] = train_mst
            extra_scalars[f'{tag}/s_mmd_both_mst'] = 0.5 * (mst + train_mst)

        if autoguidance_params is not None:
          for w_extra, gen_fn in extra_weight_generators.items():
            tag = f'ag_w{w_extra}'
            mst, sub, div, train_mst = score_draw(
                gen_fn,
                state,
                generation_rng,
                cond,
                point_cond_mask,
                class_labels_all,
                n_gen_chunks,
                n_samples_to_gen,
            )
            extra_scalars[f'{tag}/s_mmd_val_mst'] = mst
            extra_scalars[f'{tag}/s_mmd_val'] = sub
            extra_scalars[f'{tag}/s_mmd_train_mst'] = train_mst
            extra_scalars[f'{tag}/s_mmd_both_mst'] = 0.5 * (mst + train_mst)

      # Headline = the configured sampler (autoguidance_weight / t_min / t_max)
      # on the headline draw; it is autoguided whenever a guide exists
      # (`ag_active`). Variants are only logged under ev_<name>/: picking the
      # best variant as the headline would be an optimistic selection bias.
      primary_mst = s_mmd_val
      primary_sub = s_mmd_val_sub
      s_mmd_both_mst = 0.5 * (primary_mst + s_mmd_train)

      writer.write_scalars(
          step * global_batch_size,
          {
              'mmd_train': mmd_train,
              'mmd_val': mmd_val,
              'fid_train': fid_train,
              'fid_val': fid_val,
              's_mmd_train': s_mmd_train_sub,
              's_mmd_val': primary_sub,
              's_fid_train': s_fid_train,
              's_fid_val': s_fid_val,
              's_mmd_train_mst': s_mmd_train,
              's_mmd_val_mst': primary_mst,
              's_mmd_both_mst': s_mmd_both_mst,
              'ag_active': float(autoguidance_params is not None),
              'generation_time': metrics_start_time - generation_start_time,
              'metrics_time': time.time() - metrics_start_time,
          },
      )
      logging.info(
          'Step: %d, MMD Train: %.5f, MMD Val: %.5f, s_MMD Val MST: %.5f, s_MMD'
          ' Train MST: %.5f, s_MMD Both MST: %.5f, Min s_MMD Train: %.5f',
          step,
          mmd_train,
          mmd_val,
          primary_mst,
          s_mmd_train,
          s_mmd_both_mst,
          state.min_s_mmd_train,
      )
      if headline_div:
        writer.write_scalars(
            step * global_batch_size,
            {f'div_{k}': v for k, v in headline_div.items()},
        )
      if extra_scalars:
        writer.write_scalars(step * global_batch_size, extra_scalars)
      writer.flush()
      train_start_time = time.time()

      # barrier for multihost TPU support (writer) and easier debugging
      # TODO(riegerfr): consider sync_global_devices
      multihost_utils.process_allgather(jax.numpy.array(0))

      if (
          config.stop_training_mmd_threshold > 0  # allow disabling with -1
          and jax.device_get(state.min_s_mmd_train)
          > config.stop_training_mmd_threshold
          and step >= config.stop_training_min_steps
      ):
        logging.info('Stopping training at step %d because of high MMD', step)
        break

  if cooldown_start_step > 0 and cooldown_branches:
    logging.info(
        'Main training reached step %d. Starting sequential cooldown'
        ' branches: %r',
        int(jax.device_get(state.step)),
        cooldown_branches,
    )
    cooldown_base_dir = ckpt_dir / 'cooldown_base'
    cooldown_base_manager = ocp.CheckpointManager(
        directory=cooldown_base_dir,
        checkpointers={
            'train_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
            # pytype:disable=wrong-arg-types, pyrefly: ignore[bad-argument-type]
            'train_iter': ocp.Checkpointer(grain.OrbaxCheckpointHandler()),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, cleanup_tmp_directories=True
        ),
    )
    if cooldown_base_manager.latest_step() is not None:
      restored_base = cooldown_base_manager.restore(
          cooldown_base_manager.latest_step(),
          items={'train_state': state, 'train_iter': train_iter},
      )
      base_state = restored_base['train_state']
      train_iter = restored_base['train_iter']
      step_base = int(jax.device_get(base_state.step))
      logging.info(
          'Restored base state from %s (step %d) for cooldown branching.',
          cooldown_base_dir,
          step_base,
      )
    else:
      base_state = state
      step_base = int(jax.device_get(base_state.step))
      cooldown_base_manager.save(
          step_base,
          items={'train_state': base_state, 'train_iter': train_iter},
      )
      cooldown_base_manager.wait_until_finished()
      if extra_ema_manager is not None:
        extra_ema_manager.save(step_base, items={'extra_emas': extra_emas})
        extra_ema_manager.wait_until_finished()
      if posthoc_ema_manager is not None:
        posthoc_ema_manager.save(
            step_base, items={'posthoc_emas': posthoc_emas}
        )
        posthoc_ema_manager.wait_until_finished()
      logging.info(
          'Saved base state to dedicated cooldown_base dir %s at step %d.',
          cooldown_base_dir,
          step_base,
      )

    base_snapshot_ema = jax.device_put(
        jax.tree_util.tree_map(jnp.copy, base_state.ema_params),
        replicate_sharding,
    )
    branch_ema_snapshots = [base_snapshot_ema]

    cooldown_in_branch_dir = ckpt_dir / 'cooldown_in_branch'
    cooldown_in_branch_manager = ocp.CheckpointManager(
        directory=cooldown_in_branch_dir,
        checkpointers={
            'branch_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
            # pytype:disable=wrong-arg-types, pyrefly: ignore[bad-argument-type]
            'train_iter': ocp.Checkpointer(grain.OrbaxCheckpointHandler()),
            'metadata': ocp.Checkpointer(ocp.JsonCheckpointHandler()),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=2, cleanup_tmp_directories=True
        ),
    )

    final_branch_state = None
    save_checkpoint_steps = int(config.get('save_checkpoint_steps', 10000))
    last_cooldown_checkpoint_time = time.time()

    for b_idx, branch_steps in enumerate(cooldown_branches):
      branch_dir = ckpt_dir / f'cooldown_branch_{b_idx}'
      branch_mgr = ocp.CheckpointManager(
          directory=branch_dir,
          checkpointers={
              'ema_params': ocp.Checkpointer(
                  ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
              ),
              'branch_state': ocp.Checkpointer(
                  ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
              ),
          },
          options=ocp.CheckpointManagerOptions(
              max_to_keep=1, cleanup_tmp_directories=True
          ),
      )

      if branch_mgr.latest_step() is not None:
        logging.info(
            'Branch %d/%d already finished (checkpoint at %s). Restoring EMA'
            ' snapshot...',
            b_idx + 1,
            len(cooldown_branches),
            branch_dir,
        )
        restored_branch = branch_mgr.restore(
            branch_mgr.latest_step(),
            items={'ema_params': base_snapshot_ema, 'branch_state': base_state},
        )
        branch_ema = jax.device_put(
            restored_branch['ema_params'], replicate_sharding
        )
        branch_ema_snapshots.append(branch_ema)
        final_branch_state = restored_branch['branch_state']
        continue

      start_s_in_branch = 0
      branch_state = jax.tree_util.tree_map(jnp.copy, base_state)
      if cooldown_in_branch_manager.latest_step() is not None:
        try:
          in_b_data = cooldown_in_branch_manager.restore(
              cooldown_in_branch_manager.latest_step(),
              items={
                  'branch_state': branch_state,
                  'train_iter': train_iter,
                  'metadata': {},
              },
          )
          meta = in_b_data.get('metadata', {})
          if (
              meta.get('branch_idx') == b_idx
              and 0 < meta.get('s_in_branch', 0) < branch_steps
          ):
            branch_state = in_b_data['branch_state']
            train_iter = in_b_data['train_iter']
            start_s_in_branch = int(meta['s_in_branch'])
            logging.info(
                'Resuming branch %d from in-branch step %d/%d (global %d).',
                b_idx + 1,
                start_s_in_branch,
                branch_steps,
                int(jax.device_get(branch_state.step)),
            )
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              'Failed to restore in-branch checkpoint (%s), starting from'
              ' base.',
              e,
          )
          branch_state = jax.tree_util.tree_map(jnp.copy, base_state)
          start_s_in_branch = 0

      logging.info(
          'Starting cooldown branch %d/%d (%d steps, from step %d) from base'
          ' step %d...',
          b_idx + 1,
          len(cooldown_branches),
          branch_steps,
          start_s_in_branch,
          step_base,
      )
      branch_running_loss = 0.0
      b_start_time = time.time()
      for s_in_branch in range(start_s_in_branch, branch_steps):
        cooldown_scale = jnp.float32(
            max(0.0, 1.0 - s_in_branch / float(branch_steps))
        )
        batch = next(train_iter)
        batch = {
            'coord': batch['coord'],
            'feat': batch['feat'],
            '_dataset_index': (
                batch['_dataset_index'] if '_dataset_index' in batch else None
            ),
        }
        if (
            config.get('combined_batch_mode', False)
            and _slice_to_full is not None
        ):
          full_batch_np = jax.tree.map(_slice_to_full, batch)
          batch_jnp = jax.tree.map(jnp.array, batch)
          full_batch_jnp = jax.tree.map(jnp.array, full_batch_np)
          branch_state, aux = p_train_step_cooldown_combined(
              branch_state, batch_jnp, full_batch_jnp, cooldown_scale
          )
        elif config.get('cooldown_dynbatch', False):
          # As pfm7_cool: regular dynamic-batching step on the absolute step
          # grid, update scaled by the linear 1 -> 0 factor.
          batch_jnp = jax.tree.map(jnp.array, batch)
          b_step = int(jax.device_get(branch_state.step))
          b_regular = (
              (b_step % config.dynamic_batch_freq == 0)
              if config.dynamic_batch_freq > 0
              else True
          )
          branch_state, aux = p_train_step_scaled(
              branch_state, batch_jnp, b_regular, cooldown_scale
          )
        else:
          batch_jnp = jax.tree.map(jnp.array, batch)
          branch_state, aux = p_train_step_cooldown(
              branch_state, batch_jnp, cooldown_scale
          )

        if p_update_extra_emas is not None:
          extra_emas = p_update_extra_emas(
              extra_emas, branch_state.params, branch_state.step
          )
        if p_update_posthoc_emas is not None:
          posthoc_emas = p_update_posthoc_emas(
              posthoc_emas, branch_state.params, branch_state.step
          )

        branch_running_loss += aux['loss']
        cur_b_step = int(jax.device_get(branch_state.step))
        if (
            (s_in_branch + 1) % config.log_train_every_steps == 0
            or s_in_branch + 1 == branch_steps
        ):
          writer.write_scalars(
              cur_b_step * global_batch_size,
              {
                  f'branch{b_idx+1}/train_loss': (
                      branch_running_loss
                      / float((s_in_branch % config.log_train_every_steps) + 1)
                  ),
                  f'branch{b_idx+1}/lr_scale': float(cooldown_scale),
              },
          )
          branch_running_loss = 0.0
          logging.info(
              'Branch %d step %d/%d (global %d): loss=%.5f, lr_scale=%.4f'
              ' (%.2fs)',
              b_idx + 1,
              s_in_branch + 1,
              branch_steps,
              cur_b_step,
              aux['loss'],
              float(cooldown_scale),
              time.time() - b_start_time,
          )

        # Periodic in-branch checkpointing
        b_time_to_save = bool(
            save_checkpoint_secs > 0
            and (
                time.time() - last_cooldown_checkpoint_time
                >= save_checkpoint_secs
            )
        )
        b_step_to_save = bool(
            save_checkpoint_steps > 0
            and (s_in_branch + 1) % save_checkpoint_steps == 0
        )
        if (
            (b_step_to_save or b_time_to_save)
            and (s_in_branch + 1) < branch_steps
        ):
          last_cooldown_checkpoint_time = time.time()
          if b_time_to_save and jax.process_index() == 0:
            logging.info(
                'Wall-clock checkpoint save triggered in branch %d at step %d'
                ' (%.1fs elapsed).',
                b_idx + 1,
                cur_b_step,
                save_checkpoint_secs,
            )
          cooldown_in_branch_manager.save(
              cur_b_step,
              items={
                  'branch_state': branch_state,
                  'train_iter': train_iter,
                  'metadata': {
                      'branch_idx': b_idx,
                      's_in_branch': s_in_branch + 1,
                      'step': cur_b_step,
                  },
              },
          )
          cooldown_in_branch_manager.wait_until_finished()
          if extra_ema_manager is not None:
            extra_ema_manager.save(cur_b_step, items={'extra_emas': extra_emas})
            extra_ema_manager.wait_until_finished()
          if posthoc_ema_manager is not None:
            posthoc_ema_manager.save(
                cur_b_step, items={'posthoc_emas': posthoc_emas}
            )
            posthoc_ema_manager.wait_until_finished()
          logging.info(
              'Saved in-branch checkpoint for branch %d at step %d/%d (global'
              ' %d).',
              b_idx + 1,
              s_in_branch + 1,
              branch_steps,
              cur_b_step,
          )

      end_step = int(jax.device_get(branch_state.step))
      branch_ema = jax.device_put(
          jax.tree_util.tree_map(jnp.copy, branch_state.ema_params),
          replicate_sharding,
      )
      branch_ema_snapshots.append(branch_ema)
      final_branch_state = branch_state

      branch_mgr.save(
          end_step,
          items={'ema_params': branch_ema, 'branch_state': branch_state},
      )
      branch_mgr.wait_until_finished()

      latest_checkpoint_manager.save(
          end_step,
          items={'train_state': branch_state, 'train_iter': train_iter},
      )
      latest_checkpoint_manager.wait_until_finished()
      if extra_ema_manager is not None:
        extra_ema_manager.save(end_step, items={'extra_emas': extra_emas})
        extra_ema_manager.wait_until_finished()
      if posthoc_ema_manager is not None:
        posthoc_ema_manager.save(end_step, items={'posthoc_emas': posthoc_emas})
        posthoc_ema_manager.wait_until_finished()
      logging.info(
          'Completed and saved branch %d (%d steps) at global step %d.',
          b_idx + 1,
          branch_steps,
          end_step,
      )

    soup_eval_seeds = [
        int(x.strip())
        for x in str(config.get('soup_eval_seeds', '')).split(',')
        if x.strip()
    ]
    if (
        config.get('soup_final', False)
        and len(branch_ema_snapshots) == len(cooldown_branches) + 1
        and soup_eval_seeds
    ):
      # Uniform soup of [base EMA, end-of-branch EMAs] (float32 accumulation,
      # same arithmetic as `restore_model_soup` in the eval-only soup waves).
      soup_ema = jax.device_put(
          uniform_soup(branch_ema_snapshots), replicate_sharding
      )
      soup_state = (final_branch_state or state).replace(ema_params=soup_ema)
      final_step = int(jax.device_get(soup_state.step))
      soup_manager = ocp.CheckpointManager(
          directory=ckpt_dir / 'soup',
          checkpointers={
              'train_state': ocp.Checkpointer(
                  ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
              ),
          },
          options=ocp.CheckpointManagerOptions(
              max_to_keep=1, cleanup_tmp_directories=True
          ),
      )
      if soup_manager.latest_step() is None:
        soup_manager.save(
            final_step, items={'train_state': {'ema_params': soup_ema}}
        )
        soup_manager.wait_until_finished()
      logging.info(
          'Soup of %d members saved to %s at step %d.',
          len(branch_ema_snapshots),
          ckpt_dir / 'soup',
          final_step,
      )
      done_path = ckpt_dir / 'soup_eval_done.json'
      if done_path.exists():
        logging.info('Final soup eval already done (%s); skipping.', done_path)
      else:
        if ag_self_step > 0 and autoguidance_params is None:
          logging.error('Final soup eval runs WITHOUT the self-guide snapshot.')
        if init_cond is not None:
          eval_cond = jnp.concatenate(
              (jnp.zeros_like(init_cond), jnp.zeros_like(init_cond)), axis=-1
          )[: config.num_devices]
        else:
          eval_cond = None
        if config.point_cond > 0:
          eval_pcm = jnp.zeros(
              (init_coord.shape[0], init_coord.shape[1]), dtype=bool
          )[: config.num_devices]
        else:
          eval_pcm = None
        eval_n_samples = config.n_samples
        eval_n_chunks = max(1, eval_n_samples // config.num_devices)
        eval_class_labels = jax.random.bernoulli(
            jax.random.key(12345),
            (
                running_class_1_sum / running_total_count
                if running_total_count > 0
                else 0.5
            ),
            (eval_n_chunks * config.num_devices,),
        ).astype(jnp.int32)
        n_soup_draws = int(config.get('soup_eval_draws', 10))
        member_names = ['member_base'] + [
            f'member_cool{b // 1000}k' if b % 1000 == 0 else f'member_cool{b}'
            for b in cooldown_branches
        ]
        # (name, state, seed sets): the soup on every seed set, each member on
        # the first one (members are reference points, the soup is the
        # headline).
        eval_models = [('soup', soup_state, soup_eval_seeds)]
        if len(branch_ema_snapshots) == 3:
          soup_cool_ema = jax.device_put(
              uniform_soup(branch_ema_snapshots[1:]), replicate_sharding
          )
          soup_cool_state = soup_state.replace(ema_params=soup_cool_ema)
          eval_models.append(('soup_cool', soup_cool_state, soup_eval_seeds))
        eval_models += [
            (
                name,
                soup_state.replace(ema_params=member_ema),
                soup_eval_seeds[:1],
            )
            for name, member_ema in zip(member_names, branch_ema_snapshots)
        ]
        soup_scalars = {
            'soup/ag_active': float(autoguidance_params is not None),
            'soup/n_members': float(len(branch_ema_snapshots)),
        }
        soup_eval_start = time.time()
        for name, eval_state, seeds in eval_models:
          all_val, all_train, all_both = [], [], []
          for seed in seeds:
            set_val, set_train, set_both = [], [], []
            for d_idx, draw_rng in enumerate(
                paired_draw_rngs(seed, n_soup_draws)
            ):
              val_mst, _, _, train_mst = score_draw(
                  generate_chunk,
                  eval_state,
                  draw_rng,
                  eval_cond,
                  eval_pcm,
                  eval_class_labels,
                  eval_n_chunks,
                  eval_n_samples,
              )
              both_mst = 0.5 * (val_mst + train_mst)
              soup_scalars[f'{name}/s{seed}/s_mmd_val_mst_d{d_idx}'] = val_mst
              soup_scalars[f'{name}/s{seed}/s_mmd_train_mst_d{d_idx}'] = (
                  train_mst
              )
              soup_scalars[f'{name}/s{seed}/s_mmd_both_mst_d{d_idx}'] = both_mst
              set_val.append(val_mst)
              set_train.append(train_mst)
              set_both.append(both_mst)
            soup_scalars[f'{name}/s{seed}/s_mmd_val_mst_mean'] = float(
                np.mean(set_val)
            )
            soup_scalars[f'{name}/s{seed}/s_mmd_train_mst_mean'] = float(
                np.mean(set_train)
            )
            soup_scalars[f'{name}/s{seed}/s_mmd_both_mst_mean'] = float(
                np.mean(set_both)
            )
            all_val += set_val
            all_train += set_train
            all_both += set_both
          # Headline tags: mean over all paired draws of all seed sets.
          soup_scalars[f'{name}/s_mmd_val_mst'] = float(np.mean(all_val))
          soup_scalars[f'{name}/s_mmd_train_mst'] = float(np.mean(all_train))
          soup_scalars[f'{name}/s_mmd_both_mst'] = float(np.mean(all_both))
          soup_scalars[f'{name}/s_mmd_val_mst_sd'] = float(
              np.std(all_val, ddof=1) if len(all_val) > 1 else 0.0
          )
          soup_scalars[f'{name}/n_draws'] = float(len(all_val))
          logging.info(
              'Final eval %s: val %.4f train %.4f both %.4f over %d draws.',
              name,
              soup_scalars[f'{name}/s_mmd_val_mst'],
              soup_scalars[f'{name}/s_mmd_train_mst'],
              soup_scalars[f'{name}/s_mmd_both_mst'],
              len(all_val),
          )
        soup_scalars['soup/eval_time'] = time.time() - soup_eval_start
        writer.write_scalars(final_step * global_batch_size, soup_scalars)
        writer.flush()
        if jax.process_index() == 0:
          with storage.atomic_file(str(done_path), 'w') as f:
            f.write(json.dumps(soup_scalars, indent=1, sort_keys=True) + '\n')
        logging.info('Final soup eval complete: %r', soup_scalars)
    elif config.get('soup_final', False) and len(branch_ema_snapshots) == 3:
      logging.info(
          'Averaging tri-soup across [base_995k, branch1_1045k, branch2_1145k]'
          ' with 1/3 each...'
      )
      w_soup = 1.0 / 3.0

      def _soup_leaf(s0, s1, s2):
        return w_soup * (s0 + s1 + s2)

      soup_ema = jax.device_put(
          jax.tree_util.tree_map(_soup_leaf, *branch_ema_snapshots),
          replicate_sharding,
      )
      state = (final_branch_state or state).replace(ema_params=soup_ema)

      final_step = int(jax.device_get(state.step))
      best_checkpoint_manager.save(
          final_step,
          items={'train_state': state},
      )
      best_checkpoint_manager.wait_until_finished()
      logging.info('Tri-soup weights saved at step %d.', final_step)

      # Final evaluation with headline sampler on 10 paired draws x 2 seed sets
      logging.info('Running final headline sampler eval on tri-soup...')
      if init_cond is not None:
        eval_cond = jnp.concatenate(
            (jnp.zeros_like(init_cond), jnp.zeros_like(init_cond)), axis=-1
        )[: config.num_devices]
      else:
        eval_cond = None
      if config.point_cond > 0:
        eval_pcm = jnp.zeros(
            (
                init_coord.shape[0],
                init_coord.shape[1],
            ),
            dtype=bool,
        )[: config.num_devices]
      else:
        eval_pcm = None

      if config.train_set == 'pos_neg':
        eval_n_samples = 1024
        eval_class_labels = jnp.concatenate(
            [jnp.zeros(512, dtype=jnp.int32), jnp.ones(512, dtype=jnp.int32)]
        )
        eval_n_chunks = eval_n_samples // config.num_devices
      else:
        eval_n_samples = config.n_samples
        ratio = (
            running_class_1_sum / running_total_count
            if running_total_count > 0
            else 0.5
        )
        eval_n_chunks = max(1, eval_n_samples // config.num_devices)
        eval_class_labels = jax.random.bernoulli(
            jax.random.key(12345), ratio, (eval_n_chunks * config.num_devices,)
        ).astype(jnp.int32)

      soup_scalars = {}
      all_mst = []
      all_sub = []
      presoup_all_mst = []
      presoup_all_sub = []
      seed_set_1 = int(config.generation_seed)
      seed_set_2 = int(config.generation_seed) + 1000
      for s_set_idx, base_seed in [(1, seed_set_1), (2, seed_set_2)]:
        set_mst = []
        set_sub = []
        set_presoup_mst = []
        set_presoup_sub = []
        for d_idx in range(10):
          draw_rng = jax.random.fold_in(jax.random.key(base_seed), d_idx)
          mst, sub, _, _ = score_draw(
              generate_chunk,
              state,
              draw_rng,
              eval_cond,
              eval_pcm,
              eval_class_labels,
              eval_n_chunks,
              eval_n_samples,
          )
          soup_scalars[f'soup/s{s_set_idx}_d{d_idx}_mst'] = mst
          soup_scalars[f'soup/s{s_set_idx}_d{d_idx}_sub'] = sub
          set_mst.append(mst)
          set_sub.append(sub)
          all_mst.append(mst)
          all_sub.append(sub)

          if final_branch_state is not None:
            pre_mst, pre_sub, _, _ = score_draw(
                generate_chunk,
                final_branch_state,
                draw_rng,
                eval_cond,
                eval_pcm,
                eval_class_labels,
                eval_n_chunks,
                eval_n_samples,
            )
            soup_scalars[f'presoup/s{s_set_idx}_d{d_idx}_mst'] = pre_mst
            soup_scalars[f'presoup/s{s_set_idx}_d{d_idx}_sub'] = pre_sub
            set_presoup_mst.append(pre_mst)
            set_presoup_sub.append(pre_sub)
            presoup_all_mst.append(pre_mst)
            presoup_all_sub.append(pre_sub)

        soup_scalars[f'soup/s{s_set_idx}_mst_mean'] = float(np.mean(set_mst))
        soup_scalars[f'soup/s{s_set_idx}_sub_mean'] = float(np.mean(set_sub))
        if set_presoup_mst:
          soup_scalars[f'presoup/s{s_set_idx}_mst_mean'] = float(
              np.mean(set_presoup_mst)
          )
          soup_scalars[f'presoup/s{s_set_idx}_sub_mean'] = float(
              np.mean(set_presoup_sub)
          )

      soup_scalars['soup/mst_mean_20d'] = float(np.mean(all_mst))
      soup_scalars['soup/sub_mean_20d'] = float(np.mean(all_sub))
      if presoup_all_mst:
        soup_scalars['presoup/mst_mean_20d'] = float(np.mean(presoup_all_mst))
        soup_scalars['presoup/sub_mean_20d'] = float(np.mean(presoup_all_sub))
        soup_scalars['soup/delta_mst_20d'] = float(
            np.mean(all_mst) - np.mean(presoup_all_mst)
        )
        soup_scalars['soup/delta_sub_20d'] = float(
            np.mean(all_sub) - np.mean(presoup_all_sub)
        )

      writer.write_scalars(final_step * global_batch_size, soup_scalars)
      writer.flush()
      logging.info('Final tri-soup evaluation complete: %r', soup_scalars)

  writer.close()
  latest_checkpoint_manager.wait_until_finished()
  best_checkpoint_manager.wait_until_finished()
  if extra_ema_manager is not None:
    extra_ema_manager.wait_until_finished()
  if posthoc_ema_manager is not None:
    posthoc_ema_manager.wait_until_finished()
  return best_x_gen

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
"""Tests for remat-agnostic checkpoint restoring in train.py."""

import os

# Set host device count to 4 CPUs before JAX backend initializes.
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '')
    + ' --xla_force_host_platform_device_count=4'
)

from absl.testing import absltest
from absl.testing import parameterized
from connectomics.mogen.flow_matching import train
from etils import epath
import flax.core
import flax.linen as nn

utils = train.utils
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp


class _Block(nn.Module):

  @nn.compact
  def __call__(self, x):
    return nn.Dense(4)(x) + x


class _Net(nn.Module):
  remat: bool = False

  @nn.compact
  def __call__(self, x):
    block = nn.remat(_Block) if self.remat else _Block
    for _ in range(2):
      x = block()(x)
    return nn.Dense(3)(x)


def _init(remat: bool, seed: int = 0):
  model = _Net(remat=remat)
  x = jnp.ones((2, 4), jnp.float32)
  return model, model.init(jax.random.key(seed), x)['params'], x


class AlignRematNamesTest(parameterized.TestCase):

  def test_flax_remat_prefixes_module_names(self):
    _, plain, _ = _init(remat=False)
    _, remat, _ = _init(remat=True)
    self.assertIn('_Block_0', plain)
    self.assertIn('Checkpoint_Block_0', remat)

  @parameterized.named_parameters(
      ('remat_ckpt_into_plain_model', True, False),
      ('plain_ckpt_into_remat_model', False, True),
      ('plain_into_plain', False, False),
      ('remat_into_remat', True, True),
  )
  def test_cast_params_like_is_remat_agnostic(self, saved_remat, model_remat):
    _, saved, _ = _init(remat=saved_remat, seed=1)
    model, template, x = _init(remat=model_remat, seed=2)
    restored = train._cast_params_like(template, saved)
    self.assertEqual(
        jax.tree_util.tree_structure(restored),
        jax.tree_util.tree_structure(template),
    )
    ref_model = _Net(remat=saved_remat)
    np.testing.assert_allclose(
        model.apply({'params': restored}, x),
        ref_model.apply({'params': saved}, x),
        rtol=1e-6,
    )

  def test_casts_to_template_dtype(self):
    template = {'CheckpointA_0': {'k': jnp.zeros((2,), jnp.bfloat16)}}
    restored = {'A_0': {'k': np.ones((2,), np.float32)}}
    out = train._cast_params_like(template, restored)
    self.assertEqual(out['CheckpointA_0']['k'].dtype, jnp.bfloat16)

  def test_exact_keys_win_over_prefixed(self):
    # A genuine module named 'CheckpointFoo_0' next to 'Foo_0' is untouched.
    template = {'Foo_0': 1, 'CheckpointFoo_0': 2}
    restored = {'Foo_0': 10, 'CheckpointFoo_0': 20}
    self.assertEqual(
        train._align_remat_names(template, restored),
        {'Foo_0': 10, 'CheckpointFoo_0': 20},
    )

  def test_real_mismatch_still_fails(self):
    template = {'A_0': {'k': jnp.zeros(2)}}
    restored = {'B_0': {'k': np.zeros(2)}}
    self.assertEqual(train._align_remat_names(template, restored), restored)
    with self.assertRaises(ValueError):
      train._cast_params_like(template, restored)

  def test_non_mappings_pass_through(self):
    self.assertEqual(train._align_remat_names([1], (2,)), (2,))


class RestoreParamsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('remat_ckpt_into_plain_model', True, False),
      ('plain_ckpt_into_remat_model', False, True),
  )
  def test_restore_params_across_remat(self, saved_remat, model_remat):
    _, saved, x = _init(remat=saved_remat, seed=3)
    model, template, _ = _init(remat=model_remat, seed=4)
    ckpt_dir = epath.Path(self.create_tempdir().full_path)
    ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
    ).save(
        ckpt_dir / '7' / 'train_state',
        {'params': saved, 'ema_params': saved},
    )
    restored = train._restore_params(str(ckpt_dir), 7, template, use_ema=True)
    np.testing.assert_allclose(
        model.apply({'params': restored}, x),
        _Net(remat=saved_remat).apply({'params': saved}, x),
        rtol=1e-6,
    )

  def test_self_autoguidance_snapshot_round_trip(self):
    # Same manager/item layout as the self-autoguidance snapshot in
    # train_flow_matching (ckpt_dir/ag_self/<step>/train_state/ema_params).
    model, params, x = _init(remat=False, seed=5)
    _, template, _ = _init(remat=False, seed=6)
    ag_dir = epath.Path(self.create_tempdir().full_path) / 'ag_self'
    manager = ocp.CheckpointManager(
        directory=ag_dir,
        checkpointers={
            'train_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, cleanup_tmp_directories=True
        ),
    )
    self.assertIsNone(manager.latest_step())
    manager.save(20, items={'train_state': {'ema_params': params}})
    manager.wait_until_finished()
    self.assertEqual(manager.latest_step(), 20)
    restored = train._restore_params(str(ag_dir), -1, template, use_ema=True)
    np.testing.assert_allclose(
        model.apply({'params': restored}, x),
        model.apply({'params': params}, x),
        rtol=1e-6,
    )

  def test_warm_start_restore_and_continue_step(self):
    model, params, x = _init(remat=False, seed=7)
    ckpt_dir = epath.Path(self.create_tempdir().full_path) / 'warm_start_ckpt'
    manager = ocp.CheckpointManager(
        directory=ckpt_dir,
        checkpointers={
            'train_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, cleanup_tmp_directories=True
        ),
    )
    fake_state = train.utils.TrainState(
        step=995000,
        params=params,
        ema_params=params,
        batch_stats={},
        opt_state={'count': 995000},
        min_s_mmd_train=1.23,
    )
    manager.save(995000, items={'train_state': fake_state})
    manager.wait_until_finished()

    template_state = train.utils.TrainState(
        step=0,
        params=jax.tree_util.tree_map(jnp.zeros_like, params),
        ema_params=jax.tree_util.tree_map(jnp.zeros_like, params),
        batch_stats={},
        opt_state={'count': 0},
    )
    cfg = train.ml_collections.ConfigDict({
        'init_ckpt_path': str(ckpt_dir),
        'init_ckpt_step': 995000,
        'continue_step': True,
        'load_optimizer_state': True,
    })
    state_subdir = ckpt_dir / str(cfg.init_ckpt_step) / 'train_state'
    ckptr = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
    )
    restored_state = ckptr.restore(state_subdir, item=template_state)
    continue_step = cfg.get('continue_step', False) or bool(
        cfg.get('init_ckpt_path', '')
    )
    new_step = cfg.init_ckpt_step if continue_step else 0
    final_state = template_state.replace(
        step=new_step,
        params=restored_state.params,
        ema_params=restored_state.ema_params,
        batch_stats=restored_state.batch_stats,
        opt_state=restored_state.opt_state,
    )
    self.assertEqual(final_state.step, 995000)
    self.assertEqual(final_state.opt_state, {'count': 995000})
    np.testing.assert_allclose(
        model.apply({'params': final_state.params}, x),
        model.apply({'params': params}, x),
        rtol=1e-6,
    )


class ParseEvalVariantsTest(absltest.TestCase):

  def test_parse_key_value_spec(self):
    variants = train.parse_eval_variants(
        'tmin=0.2+w=2.0:tmax=0.7', has_autoguidance=True
    )
    self.assertEqual(len(variants), 2)
    self.assertEqual(variants[0][0], 'tmin0.2')
    self.assertEqual(variants[0][1], {'tmin': 0.2})
    self.assertEqual(variants[1][0], 'w2.0_tmax0.7')
    self.assertEqual(variants[1][1], {'w': 2.0, 'tmax': 0.7})

  def test_parse_temp_spec(self):
    variants = train.parse_eval_variants(
        'temp=0.9+temp=0.95:steps=200', has_autoguidance=False
    )
    self.assertEqual(
        variants,
        [
            ('temp0.9', {'temp': 0.9}),
            ('temp0.95_steps200', {'temp': 0.95, 'steps': 200}),
        ],
    )

  def test_parse_positional_spec(self):
    spec = (
        'ag150_t80:euler:100:0.0:1.5:0.0:0.8,'
        'ag165_t75:euler:100:0.0:1.65:0.0:0.75'
    )
    variants = train.parse_eval_variants(spec, has_autoguidance=True)
    self.assertEqual(len(variants), 2)
    self.assertEqual(variants[0][0], 'ag150_t80')
    self.assertEqual(
        variants[0][1],
        {
            'solver': 'euler',
            'steps': 100,
            'w': 1.5,
            'tmin': 0.0,
            'tmax': 0.8,
        },
    )
    self.assertEqual(variants[1][0], 'ag165_t75')
    self.assertEqual(
        variants[1][1],
        {
            'solver': 'euler',
            'steps': 100,
            'w': 1.65,
            'tmin': 0.0,
            'tmax': 0.75,
        },
    )

  def test_parse_pfm16_specs_and_aliases(self):
    v1 = 'w=1.4+w=1.6+w=1.5:solver=heun+w=1.5:solver=rk4'
    parsed1 = train.parse_eval_variants(v1, has_autoguidance=True)
    self.assertEqual(len(parsed1), 4)
    self.assertEqual(parsed1[0], ('w1.4', {'w': 1.4}))
    self.assertEqual(parsed1[1], ('w1.6', {'w': 1.6}))
    self.assertEqual(
        parsed1[2], ('w1.5_solverheun', {'w': 1.5, 'solver': 'heun'})
    )
    self.assertEqual(
        parsed1[3], ('w1.5_solverrk4', {'w': 1.5, 'solver': 'rk4'})
    )

    v2 = (
        'solver=heun+solver=rk4+sched=cosine_1.5+sched=linear+'
        'steps=200:solver=euler'
    )
    parsed2 = train.parse_eval_variants(v2, has_autoguidance=True)
    self.assertEqual(len(parsed2), 5)
    self.assertEqual(parsed2[0], ('solverheun', {'solver': 'heun'}))
    self.assertEqual(parsed2[1], ('solverrk4', {'solver': 'rk4'}))
    self.assertEqual(parsed2[2], ('schedcosine_1.5', {'sched': 'cosine_1.5'}))
    self.assertEqual(parsed2[3], ('schedlinear', {'sched': 'linear'}))
    self.assertEqual(
        parsed2[4], ('steps200_solvereuler', {'steps': 200, 'solver': 'euler'})
    )

    # Test aliases: schedule, t_min, t_max, step, n_steps
    v3 = 'schedule=linear+t_min=0.1+t_max=0.8+step=50+n_steps=25'
    parsed3 = train.parse_eval_variants(v3, has_autoguidance=True)
    self.assertEqual(len(parsed3), 5)
    self.assertEqual(parsed3[0], ('schedulelinear', {'sched': 'linear'}))
    self.assertEqual(parsed3[1], ('t_min0.1', {'tmin': 0.1}))
    self.assertEqual(parsed3[2], ('t_max0.8', {'tmax': 0.8}))
    self.assertEqual(parsed3[3], ('step50', {'steps': 50}))
    self.assertEqual(parsed3[4], ('n_steps25', {'steps': 25}))

  def test_parse_pfm17_specs(self):
    v0 = 'tmin=0.05+tmin=0.1+w=2.2:tmin=0.1+w=2.5:tmin=0.1'
    parsed0 = train.parse_eval_variants(v0, has_autoguidance=True)
    self.assertEqual(len(parsed0), 4)
    self.assertEqual(parsed0[0], ('tmin0.05', {'tmin': 0.05}))
    self.assertEqual(parsed0[1], ('tmin0.1', {'tmin': 0.1}))
    self.assertEqual(parsed0[2], ('w2.2_tmin0.1', {'w': 2.2, 'tmin': 0.1}))
    self.assertEqual(parsed0[3], ('w2.5_tmin0.1', {'w': 2.5, 'tmin': 0.1}))

    v1 = 'w=1.8:tmin=0.1+w=2.2:tmin=0.1+tmin=0'
    parsed1 = train.parse_eval_variants(v1, has_autoguidance=True)
    self.assertEqual(len(parsed1), 3)
    self.assertEqual(parsed1[0], ('w1.8_tmin0.1', {'w': 1.8, 'tmin': 0.1}))
    self.assertEqual(parsed1[1], ('w2.2_tmin0.1', {'w': 2.2, 'tmin': 0.1}))
    self.assertEqual(parsed1[2], ('tmin0', {'tmin': 0.0}))

    v2 = 'solver=heun+solver=rk4+sched=cosine_1.5+sched=linear'
    parsed2 = train.parse_eval_variants(v2, has_autoguidance=True)
    self.assertEqual(len(parsed2), 4)
    self.assertEqual(parsed2[0], ('solverheun', {'solver': 'heun'}))
    self.assertEqual(parsed2[1], ('solverrk4', {'solver': 'rk4'}))
    self.assertEqual(parsed2[2], ('schedcosine_1.5', {'sched': 'cosine_1.5'}))
    self.assertEqual(parsed2[3], ('schedlinear', {'sched': 'linear'}))


class ModelSoupTest(parameterized.TestCase):

  def test_weights_one_zero_reproduces_checkpoint_a_exactly(self):
    _, saved_a, _ = _init(remat=False, seed=10)
    _, saved_b, _ = _init(remat=False, seed=20)
    model, template, x = _init(remat=False, seed=30)
    ckpt_dir = epath.Path(self.create_tempdir().full_path)
    handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
    ocp.Checkpointer(handler).save(
        ckpt_dir / '100' / 'train_state',
        {'params': saved_a, 'ema_params': saved_a},
    )
    ocp.Checkpointer(handler).save(
        ckpt_dir / '200' / 'train_state',
        {'params': saved_b, 'ema_params': saved_b},
    )

    soup = train.restore_model_soup(
        [str(ckpt_dir / '100'), str(ckpt_dir / '200')],
        [1.0, 0.0],
        template_params=template,
    )
    jax.tree_util.tree_map(np.testing.assert_array_equal, soup, saved_a)
    np.testing.assert_allclose(
        model.apply({'params': soup}, x),
        _Net(remat=False).apply({'params': saved_a}, x),
        rtol=1e-6,
    )

  def test_weights_half_half_equals_manual_average(self):
    _, saved_a, _ = _init(remat=False, seed=11)
    _, saved_b, _ = _init(remat=False, seed=21)
    model, template, x = _init(remat=False, seed=31)
    ckpt_dir = epath.Path(self.create_tempdir().full_path)
    handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
    ocp.Checkpointer(handler).save(
        ckpt_dir / '100' / 'train_state',
        {'params': saved_a, 'ema_params': saved_a},
    )
    ocp.Checkpointer(handler).save(
        ckpt_dir / '200' / 'train_state',
        {'params': saved_b, 'ema_params': saved_b},
    )

    soup = train.restore_model_soup(
        [str(ckpt_dir / '100'), str(ckpt_dir / '200')],
        [0.5, 0.5],
        template_params=template,
    )
    expected_manual = jax.tree_util.tree_map(
        lambda a, b: (a + b) / 2.0, saved_a, saved_b
    )
    jax.tree_util.tree_map(np.testing.assert_allclose, soup, expected_manual)
    np.testing.assert_allclose(
        model.apply({'params': soup}, x),
        _Net(remat=False).apply({'params': expected_manual}, x),
        rtol=1e-6,
    )

  def test_mismatched_tree_structure_raises(self):
    tree_a = {'layer1': jnp.ones((2, 2)), 'layer2': jnp.zeros((2, 2))}
    tree_b = {'layer1': jnp.ones((2, 2)), 'layer3': jnp.zeros((2, 2))}
    with self.assertRaises(ValueError):
      train.average_pytrees([tree_a, tree_b], [0.5, 0.5])

  def test_mismatched_leaf_shape_raises(self):
    tree_a = {'layer1': jnp.ones((2, 2))}
    tree_b = {'layer1': jnp.ones((2, 3))}
    with self.assertRaises(ValueError):
      train.average_pytrees([tree_a, tree_b], [0.5, 0.5])

  def test_weights_sum_not_one_raises(self):
    tree_a = {'w': jnp.ones(2)}
    tree_b = {'w': jnp.ones(2)}
    with self.assertRaises(ValueError):
      train.average_pytrees([tree_a, tree_b], [0.5, 0.4])

  def test_weights_length_mismatch_raises(self):
    tree_a = {'w': jnp.ones(2)}
    tree_b = {'w': jnp.ones(2)}
    with self.assertRaises(ValueError):
      train.average_pytrees([tree_a, tree_b], [1.0])


def _warmup(step):
  return (1.0 + step) / (10.0 + step)


class SotaOneRunHelpersTest(parameterized.TestCase):
  """Helpers of the single-run SOTA recipe (pfm27)."""

  def test_polyak_switch_disabled_is_identity(self):
    for step in (0, 5, 560000, 10**6):
      self.assertIs(train.effective_polyak_decay(step, 0.999), 0.999)
      self.assertIs(train.effective_polyak_decay(step, 0.999, 0, 0.9999), 0.999)

  def test_polyak_switch_values(self):
    f = lambda s: float(train.effective_polyak_decay(s, 0.999, 560000, 0.9999))
    np.testing.assert_allclose(f(0), 0.999, rtol=1e-6)
    np.testing.assert_allclose(f(559999), 0.999, rtol=1e-6)
    np.testing.assert_allclose(f(560000), 0.1, rtol=1e-6)  # warm-up restarts
    np.testing.assert_allclose(f(560090), 91.0 / 100.0, rtol=1e-6)
    np.testing.assert_allclose(f(1555000), 0.9999, rtol=1e-6)
    traced = jax.jit(
        lambda s: train.effective_polyak_decay(s, 0.999, 560000, 0.9999)
    )(jnp.int32(560090))
    np.testing.assert_allclose(float(traced), 0.91, rtol=1e-6)

  def test_polyak_switch_reproduces_step_reset_warm_start(self):
    # Stage A: EMA 0.999 on steps [0, S); stage B restarts the step counter at
    # S with EMA 0.9999. `update_state` applies min(decay, warmup(step)).
    switch, n = 50, 120
    rng = np.random.default_rng(0)
    params = rng.normal(size=(n + 1, 4)).astype(np.float32)
    ema_ref = params[0].copy()
    ema_one = jnp.asarray(params[0])
    for step in range(n):
      if step < switch:
        d_ref = min(0.999, _warmup(step))
      else:
        d_ref = min(0.9999, _warmup(step - switch))
      ema_ref = d_ref * ema_ref + (1 - d_ref) * params[step + 1]
      d_one = train.effective_polyak_decay(step, 0.999, switch, 0.9999)
      ema_one = train.ema_update(
          ema_one, jnp.asarray(params[step + 1]), step, d_one
      )
    np.testing.assert_allclose(np.asarray(ema_one), ema_ref, rtol=1e-5)

  def test_guide_ema_update_matches_update_state_arithmetic(self):
    # Same formula as utils.update_state (decay < 1 branch).
    old = {'a': jnp.arange(3.0), 'b': {'c': jnp.ones((2, 2))}}
    new = {'a': jnp.arange(3.0) + 5.0, 'b': {'c': jnp.zeros((2, 2))}}
    for step, decay in ((0, 0.9999), (7, 0.9999), (100000, 0.9999), (3, 0.05)):
      d = min(decay, _warmup(step))
      got = train.ema_update(old, new, jnp.int32(step), decay)
      want = jax.tree_util.tree_map(
          lambda o, n, d=d: d * o + (1 - d) * n, old, new
      )
      jax.tree_util.tree_map(
          lambda g, w: np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-6),
          got,
          want,
      )

  def test_guide_ema_differs_from_main_ema(self):
    # EMA 0.999 and 0.9999 are identical while the (1 + s) / (10 + s) warm-up
    # is below 0.999 (s <= 8990) and diverge afterwards, so the 100k guide
    # needs its own EMA. Vectorized: the EMA weight of each param step.
    def ema_weights(decay, n):
      d = np.minimum(decay, _warmup(np.arange(n, dtype=np.float64)))
      # w_i = (1 - d_i) * prod_{j > i} d_j
      tail = np.concatenate([np.cumprod(d[::-1])[::-1][1:], [1.0]])
      return (1 - d) * tail

    np.testing.assert_allclose(
        ema_weights(0.999, 8000), ema_weights(0.9999, 8000), rtol=1e-12
    )
    w_main, w_guide = ema_weights(0.999, 100000), ema_weights(0.9999, 100000)
    self.assertGreater(np.abs(w_main - w_guide).sum(), 0.5)
    # Cross-check the closed form against `ema_update` on a short horizon.
    p = np.random.default_rng(1).normal(size=(9100, 2)).astype(np.float32)
    ema = jnp.zeros(2)
    for step in range(9100):
      ema = train.ema_update(ema, p[step], step, 0.999)
    np.testing.assert_allclose(
        ema, ema_weights(0.999, 9100) @ p, rtol=1e-3, atol=1e-4
    )

  def test_uniform_soup_equals_manual_mean(self):
    trees = [
        {
            'w': jnp.full((2, 3), float(i)),
            'b': jnp.full((3,), i * 2.0, jnp.bfloat16),
        }
        for i in (1, 2, 6)
    ]
    soup = train.uniform_soup(trees)
    np.testing.assert_allclose(soup['w'], np.full((2, 3), 3.0), rtol=1e-6)
    self.assertEqual(soup['b'].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(soup['b'], np.float32), np.full((3,), 6.0), rtol=1e-2
    )
    np.testing.assert_allclose(
        train.uniform_soup([{'w': jnp.ones(2)}, {'w': jnp.zeros(2)}])['w'],
        [0.5, 0.5],
    )

  def test_paired_draw_rngs_match_eval_only_protocol(self):
    seed, step = 1617, 10
    keys = train.paired_draw_rngs(seed, 3, eval_step=step)
    self.assertLen(keys, 3)
    # Headline draw: generation_rng after one split (first eval of the run).
    _, generation_rng = jax.random.split(jax.random.key(seed))
    np.testing.assert_array_equal(
        jax.random.key_data(keys[0]), jax.random.key_data(generation_rng)
    )
    # Extra draw k: fold_in(fold_in(key(seed), 1_000_003 + k), step).
    for k in (1, 2):
      want = jax.random.fold_in(
          jax.random.fold_in(jax.random.key(seed), 1_000_003 + k), step
      )
      np.testing.assert_array_equal(
          jax.random.key_data(keys[k]), jax.random.key_data(want)
      )
    self.assertFalse(
        np.array_equal(
            jax.random.key_data(train.paired_draw_rngs(1819, 1)[0]),
            jax.random.key_data(keys[0]),
        )
    )


class _ToyFMModel(nn.Module):

  @nn.compact
  def __call__(self, coord, t, **kwargs):
    c = jnp.mean(coord, axis=1, keepdims=True)
    r = coord - c
    scale = self.param('radial_scale', nn.initializers.ones, (1,))
    return nn.Dense(3)(coord) + jnp.sin(t[:, None, None]) + (scale - 1.0) * r


class _ExactVelocityModel(nn.Module):
  mu: float
  s: float

  @nn.compact
  def __call__(self, coord, t, **kwargs):
    t_exp = jnp.asarray(t)[:, None, None]
    sigma_t_sq = (1.0 - t_exp) ** 2 + (t_exp**2) * (self.s**2)
    return (
        (1.0 - t_exp) * self.mu + coord * (t_exp * (self.s**2) - (1.0 - t_exp))
    ) / sigma_t_sq


class ChurnSamplerTest(parameterized.TestCase):

  def test_parse_churn_variants(self):
    """(d) Variant parsing for churn=0.1 and churn=0.2:ctmax=0.8."""
    parsed1 = train.parse_eval_variants('churn=0.1', has_autoguidance=False)
    self.assertEqual(parsed1, [('churn0.1', {'churn': 0.1})])

    parsed2 = train.parse_eval_variants(
        'churn=0.2:ctmax=0.8', has_autoguidance=False
    )
    self.assertEqual(
        parsed2, [('churn0.2_ctmax0.8', {'churn': 0.2, 'ctmax': 0.8})]
    )

  def test_churn_zero_bit_identical(self):
    """(a) churn=0 bit-identical output for midpoint and with autoguidance."""
    model = _ToyFMModel()
    key = jax.random.key(0)
    params = model.init(key, coord=jnp.ones((2, 4, 3)), t=jnp.zeros((2,)))[
        'params'
    ]
    state = utils.TrainState(
        step=0,
        params=params,
        ema_params=params,
        batch_stats=flax.core.FrozenDict(),
        opt_state=None,
    )

    # Midpoint without autoguidance: default vs churn=0.0
    out_default = utils.generate_samples(
        model,
        state,
        2,
        (4, 3),
        jax.random.key(123),
        n_steps=10,
        solver='midpoint',
    )
    out_churn0 = utils.generate_samples(
        model,
        state,
        2,
        (4, 3),
        jax.random.key(123),
        n_steps=10,
        solver='midpoint',
        churn=0.0,
    )
    np.testing.assert_array_equal(out_default, out_churn0)

    # Midpoint with autoguidance: default vs churn=0.0
    ag_params = jax.tree_util.tree_map(lambda x: x * 0.5, params)
    out_ag_default = utils.generate_samples(
        model,
        state,
        2,
        (4, 3),
        jax.random.key(123),
        n_steps=10,
        solver='midpoint',
        autoguidance_params=ag_params,
        autoguidance_weight=1.5,
    )
    out_ag_churn0 = utils.generate_samples(
        model,
        state,
        2,
        (4, 3),
        jax.random.key(123),
        n_steps=10,
        solver='midpoint',
        autoguidance_params=ag_params,
        autoguidance_weight=1.5,
        churn=0.0,
    )
    np.testing.assert_array_equal(out_ag_default, out_ag_churn0)

  def test_monte_carlo_renoising(self):
    """(b) Monte Carlo: renoising x_t built from fixed x1 gives mean t_hat*x1

    and std (1-t_hat).
    """
    t_i = 0.4
    churn = 0.2
    x1 = jnp.array([2.5, -1.5, 0.5], dtype=jnp.float32)
    n_samples = 200_000
    rng = jax.random.key(42)
    k0, kz = jax.random.split(rng)
    x0 = jax.random.normal(k0, (n_samples, 3))
    xt = (1.0 - t_i) * x0 + t_i * x1

    sigma = (1.0 - t_i) / t_i
    sigma_hat = sigma * (1.0 + churn)
    t_hat = 1.0 / (1.0 + sigma_hat)
    a = t_hat / t_i
    b = jnp.sqrt((1.0 - t_hat) ** 2 - (a * (1.0 - t_i)) ** 2)

    z = jax.random.normal(kz, (n_samples, 3))
    x_hat = a * xt + b * z

    emp_mean = jnp.mean(x_hat, axis=0)
    emp_std = jnp.std(x_hat, axis=0)
    exp_mean = t_hat * x1
    exp_std = 1.0 - t_hat

    np.testing.assert_allclose(emp_mean, exp_mean, atol=0.015)
    np.testing.assert_allclose(emp_std, exp_std, atol=0.015)

  def test_gaussian_1d_toy_exact_velocity(self):
    """(c) 1-D Gaussian toy x1~N(mu, s^2): ODE and churn samplers match mu

    and s.
    """
    mu = 3.0
    s = 1.5
    model = _ExactVelocityModel(mu=mu, s=s)
    key = jax.random.key(0)
    params = model.init(key, coord=jnp.ones((2, 1, 1)), t=jnp.zeros((2,))).get(
        'params', flax.core.FrozenDict()
    )
    state = utils.TrainState(
        step=0,
        params=params,
        ema_params=params,
        batch_stats=flax.core.FrozenDict(),
        opt_state=None,
    )

    n_samples = 60_000
    # ODE sampler (churn=0)
    x_ode = utils.generate_samples(
        model,
        state,
        n_samples=n_samples,
        sample_shape=(1, 1),
        rng=jax.random.key(123),
        n_steps=50,
        schedule='linear',
        solver='midpoint',
        churn=0.0,
    )
    np.testing.assert_allclose(float(jnp.mean(x_ode)), mu, atol=0.05)
    np.testing.assert_allclose(float(jnp.std(x_ode)), s, atol=0.05)

    # Churn sampler (churn=0.2)
    x_churn = utils.generate_samples(
        model,
        state,
        n_samples=n_samples,
        sample_shape=(1, 1),
        rng=jax.random.key(123),
        n_steps=50,
        schedule='linear',
        solver='midpoint',
        churn=0.2,
    )
    np.testing.assert_allclose(float(jnp.mean(x_churn)), mu, atol=0.05)
    np.testing.assert_allclose(float(jnp.std(x_churn)), s, atol=0.05)

  def test_scale_decoupled_autoguidance_mean_max_preservation(self):
    """Verifies scale-decoupled autoguidance preserves radial scale/mean_max."""
    model = _ToyFMModel()
    coord = jax.random.normal(jax.random.key(1), (4, 64, 3))
    t = jnp.array([0.5, 0.5, 0.5, 0.5])
    p_good = model.init(jax.random.key(0), coord=coord, t=t)['params']
    p_bad = (
        flax.core.unfreeze(p_good)
        if hasattr(p_good, 'unfreeze')
        else dict(p_good)
    )
    p_bad['radial_scale'] = jnp.array([0.5], dtype=jnp.float32)
    p_bad = flax.core.freeze(p_bad)
    state = utils.TrainState(
        step=0,
        params=p_good,
        ema_params=p_good,
        batch_stats=flax.core.FrozenDict(),
        opt_state=None,
    )
    v_fn_good = utils.make_velocity_fn(model, state)
    v_fn_decoupled = utils.make_velocity_fn(
        model,
        state,
        autoguidance_params=p_bad,
        autoguidance_weight=2.5,
        ag_decouple_scale=True,
    )
    v_fn_coupled = utils.make_velocity_fn(
        model,
        state,
        autoguidance_params=p_bad,
        autoguidance_weight=2.5,
        ag_decouple_scale=False,
    )
    v_good = v_fn_good(coord, t)
    v_dec = v_fn_decoupled(coord, t)
    v_coup = v_fn_coupled(coord, t)

    # 1. Check orthogonality of guidance velocity to radial direction r
    c = jnp.mean(coord, axis=1, keepdims=True)
    r = coord - c
    delta_v_dec = v_dec - v_good
    delta_v_dec_c = delta_v_dec - jnp.mean(delta_v_dec, axis=1, keepdims=True)
    radial_proj_dec = jnp.sum(r * delta_v_dec_c, axis=(1, 2))
    self.assertLess(float(jnp.max(jnp.abs(radial_proj_dec))), 1e-4)

    delta_v_coup = v_coup - v_good
    delta_v_coup_c = delta_v_coup - jnp.mean(
        delta_v_coup, axis=1, keepdims=True
    )
    radial_proj_coup = jnp.sum(r * delta_v_coup_c, axis=(1, 2))
    self.assertGreater(float(jnp.max(jnp.abs(radial_proj_coup))), 1e-3)

    # 2. Check mean_max distance preservation after an Euler step
    dt = 0.05
    step_good = coord + dt * v_good
    step_dec = coord + dt * v_dec
    step_coup = coord + dt * v_coup

    def mean_max_dist(x):
      diff = x[:, :, None, :] - x[:, None, :, :]
      dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
      return jnp.mean(jnp.max(dists, axis=-1))

    mm_good = float(mean_max_dist(step_good))
    mm_dec = float(mean_max_dist(step_dec))
    mm_coup = float(mean_max_dist(step_coup))

    self.assertAlmostEqual(mm_dec, mm_good, delta=0.01 * mm_good)
    self.assertGreater(abs(mm_coup - mm_good), abs(mm_dec - mm_good) * 3)

  def test_posthoc_ema_reconstruction(self):
    """Verifies EDM2 post-hoc EMA reconstruction formula and convex

    combination.
    """
    model = _ToyFMModel()
    key = jax.random.key(0)
    p1 = model.init(key, coord=jnp.ones((2, 4, 3)), t=jnp.zeros((2,)))['params']
    p2 = jax.tree_util.tree_map(lambda x: x * 2.0, p1)
    posthoc_emas = {'ema_gamma1': p1, 'ema_gamma2': p2}

    rec1 = utils.reconstruct_posthoc_ema(posthoc_emas, 1.0)
    np.testing.assert_allclose(
        rec1['Dense_0']['kernel'], p1['Dense_0']['kernel'], atol=1e-6
    )

    rec0 = utils.reconstruct_posthoc_ema(posthoc_emas, 0.0)
    np.testing.assert_allclose(
        rec0['Dense_0']['kernel'], p2['Dense_0']['kernel'], atol=1e-6
    )

    rec04 = utils.reconstruct_posthoc_ema(posthoc_emas, 0.4)
    expected = jax.tree_util.tree_map(lambda x: 1.6 * x, p1)
    np.testing.assert_allclose(
        rec04['Dense_0']['kernel'], expected['Dense_0']['kernel'], atol=1e-6
    )

    rec_tuple = utils.reconstruct_posthoc_ema((p1, p2), 0.4)
    np.testing.assert_allclose(
        rec_tuple['Dense_0']['kernel'], expected['Dense_0']['kernel'], atol=1e-6
    )

  def test_guard_16_gpu(self):
    """Verifies 16-GPU guard fails fast when device count does not match

    expected.
    """
    train.guard_16_gpu(expected=16, enabled=False)

    cur_devs = len(jax.devices())
    train.guard_16_gpu(expected=cur_devs, enabled=True)

    with self.assertRaises(RuntimeError) as ctx:
      train.guard_16_gpu(expected=cur_devs + 1, enabled=True)
    self.assertIn('16-GPU Guard', str(ctx.exception))


class CrossMeshRestoreTest(parameterized.TestCase):
  """Tests saving on one mesh and restoring on a different mesh / device

  count.
  """

  def setUp(self):
    super().setUp()
    self.devices = jax.devices()

  def test_cross_mesh_restore_device_count_change(self):
    self.assertGreaterEqual(
        len(self.devices),
        2,
        'Need >= 2 devices for cross-mesh restore test, got'
        f' {len(self.devices)}',
    )
    # Save on Mesh 1: 2 devices (e.g. devices[0:2])
    mesh_save = jax.sharding.Mesh(np.array(self.devices[:2]), ('batch',))
    sharding_save = jax.sharding.NamedSharding(
        mesh_save, jax.sharding.PartitionSpec()
    )

    model, params, x = _init(remat=False, seed=42)
    fake_state = train.utils.TrainState(
        step=1234,
        params=params,
        ema_params=params,
        batch_stats={},
        opt_state={'count': 1234, 'opt': jax.tree.map(jnp.zeros_like, params)},
        min_s_mmd_train=0.5,
    )
    state_saved = jax.device_put(fake_state, sharding_save)

    ckpt_dir = epath.Path(self.create_tempdir().full_path) / 'mesh_ckpt'
    save_mgr = ocp.CheckpointManager(
        directory=ckpt_dir,
        checkpointers={
            'train_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
    )
    save_mgr.save(1234, items={'train_state': state_saved})
    save_mgr.wait_until_finished()

    # Target mesh on different device count: 1 device (devices[0:1])
    mesh_1dev = jax.sharding.Mesh(np.array(self.devices[:1]), ('batch',))
    sharding_1dev = jax.sharding.NamedSharding(
        mesh_1dev, jax.sharding.PartitionSpec()
    )
    template_state_1dev = jax.device_put(
        train.utils.TrainState(
            step=0,
            params=jax.tree.map(jnp.zeros_like, params),
            ema_params=jax.tree.map(jnp.zeros_like, params),
            batch_stats={},
            opt_state={'count': 0, 'opt': jax.tree.map(jnp.zeros_like, params)},
            min_s_mmd_train=float('inf'),
        ),
        sharding_1dev,
    )

    # Restoring on mesh_1dev with make_restore_args succeeds
    restore_args_1dev = train.make_restore_args(
        template_state_1dev, sharding_1dev
    )
    restore_mgr = ocp.CheckpointManager(
        directory=ckpt_dir,
        checkpointers={
            'train_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
    )
    restored_1dev = restore_mgr.restore(
        1234,
        items={'train_state': template_state_1dev},
        restore_kwargs={'train_state': {'restore_args': restore_args_1dev}},
    )['train_state']

    self.assertEqual(restored_1dev.step, 1234)
    self.assertEqual(restored_1dev.opt_state['count'], 1234)
    np.testing.assert_allclose(
        model.apply({'params': restored_1dev.params}, x),
        model.apply({'params': params}, x),
        rtol=1e-5,
    )
    for leaf in jax.tree.leaves(restored_1dev.params):
      self.assertEqual(leaf.sharding, sharding_1dev)

    # Also restore onto a 4-device mesh (or 2 vs 4 device transition) if
    # available
    if len(self.devices) >= 4:
      mesh_4dev = jax.sharding.Mesh(np.array(self.devices[:4]), ('batch',))
      sharding_4dev = jax.sharding.NamedSharding(
          mesh_4dev, jax.sharding.PartitionSpec()
      )
      template_4dev = jax.device_put(template_state_1dev, sharding_4dev)
      restore_args_4dev = train.make_restore_args(template_4dev, sharding_4dev)
      restored_4dev = restore_mgr.restore(
          1234,
          items={'train_state': template_4dev},
          restore_kwargs={'train_state': {'restore_args': restore_args_4dev}},
      )['train_state']
      for leaf in jax.tree.leaves(restored_4dev.params):
        self.assertEqual(leaf.sharding, sharding_4dev)
      np.testing.assert_allclose(
          model.apply({'params': restored_4dev.params}, x),
          model.apply({'params': params}, x),
          rtol=1e-5,
      )

      # Disjoint device IDs: saved on devices[:2], restore on devices[2:4]
      mesh_disjoint = jax.sharding.Mesh(np.array(self.devices[2:4]), ('batch',))
      sharding_disjoint = jax.sharding.NamedSharding(
          mesh_disjoint, jax.sharding.PartitionSpec()
      )
      template_disjoint = jax.device_put(template_state_1dev, sharding_disjoint)
      restore_args_disjoint = train.make_restore_args(
          template_disjoint, sharding_disjoint
      )
      restored_disjoint = restore_mgr.restore(
          1234,
          items={'train_state': template_disjoint},
          restore_kwargs={
              'train_state': {'restore_args': restore_args_disjoint}
          },
      )['train_state']
      for leaf in jax.tree.leaves(restored_disjoint.params):
        self.assertEqual(leaf.sharding, sharding_disjoint)
      np.testing.assert_allclose(
          model.apply({'params': restored_disjoint.params}, x),
          model.apply({'params': params}, x),
          rtol=1e-5,
      )

  def test_guide_ema_cross_mesh_restore(self):
    self.assertGreaterEqual(len(self.devices), 2)
    mesh_2dev = jax.sharding.Mesh(np.array(self.devices[:2]), ('batch',))
    sharding_2dev = jax.sharding.NamedSharding(
        mesh_2dev, jax.sharding.PartitionSpec()
    )
    mesh_1dev = jax.sharding.Mesh(np.array(self.devices[:1]), ('batch',))
    sharding_1dev = jax.sharding.NamedSharding(
        mesh_1dev, jax.sharding.PartitionSpec()
    )

    _, params, _ = _init(remat=False, seed=43)
    guide_saved = jax.device_put(params, sharding_2dev)

    ckpt_dir = epath.Path(self.create_tempdir().full_path) / 'guide_mesh_ckpt'
    mgr = ocp.CheckpointManager(
        directory=ckpt_dir,
        checkpointers={
            'guide_ema': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
    )
    mgr.save(50, items={'guide_ema': guide_saved})
    mgr.wait_until_finished()

    guide_restore_args = train.make_restore_args(params, sharding_1dev)
    restored = mgr.restore(
        50,
        items={'guide_ema': params},
        restore_kwargs={'guide_ema': {'restore_args': guide_restore_args}},
    )['guide_ema']
    for leaf in jax.tree.leaves(restored):
      self.assertEqual(leaf.sharding, sharding_1dev)
    np.testing.assert_allclose(
        restored['Dense_0']['kernel'], params['Dense_0']['kernel']
    )


class CheckpointCadenceTest(absltest.TestCase):
  """Tests checkpoint save cadence and wall-clock trigger logic."""

  def test_should_save_checkpoint_step_cadence(self):
    should_save, is_wc = train.should_save_checkpoint(
        step=10000,
        save_checkpoint_steps=10000,
        last_checkpoint_time=100.0,
        save_checkpoint_secs=900,
        now=200.0,
        max_steps=100000,
    )
    self.assertTrue(should_save)
    self.assertFalse(is_wc)

  def test_should_save_checkpoint_wall_clock_trigger(self):
    # Fires when wall-clock interval elapsed even if step is not on cadence.
    should_save, is_wc = train.should_save_checkpoint(
        step=1234,
        save_checkpoint_steps=10000,
        last_checkpoint_time=100.0,
        save_checkpoint_secs=900,
        now=1001.0,
        max_steps=100000,
    )
    self.assertTrue(should_save)
    self.assertTrue(is_wc)

  def test_should_save_checkpoint_neither_step_nor_time(self):
    should_save, is_wc = train.should_save_checkpoint(
        step=1234,
        save_checkpoint_steps=10000,
        last_checkpoint_time=100.0,
        save_checkpoint_secs=900,
        now=500.0,
        max_steps=100000,
    )
    self.assertFalse(should_save)
    self.assertFalse(is_wc)

  def test_should_save_checkpoint_disabled_wall_clock(self):
    # Default-off (save_checkpoint_secs=0) does not trigger on time.
    should_save, is_wc = train.should_save_checkpoint(
        step=1234,
        save_checkpoint_steps=10000,
        last_checkpoint_time=100.0,
        save_checkpoint_secs=0,
        now=100000.0,
        max_steps=100000,
    )
    self.assertFalse(should_save)
    self.assertFalse(is_wc)

  def test_should_save_checkpoint_boundary_and_warm_sanity(self):
    for st in (10, 20, 100000):
      should_save, is_wc = train.should_save_checkpoint(
          step=st,
          save_checkpoint_steps=10000,
          last_checkpoint_time=100.0,
          save_checkpoint_secs=0,
          now=101.0,
          max_steps=100000,
      )
      self.assertTrue(should_save)
      self.assertFalse(is_wc)

    should_save, is_wc = train.should_save_checkpoint(
        step=3,
        save_checkpoint_steps=10000,
        last_checkpoint_time=100.0,
        save_checkpoint_secs=0,
        now=101.0,
        max_steps=100000,
        warm_sanity=True,
    )
    self.assertTrue(should_save)
    self.assertFalse(is_wc)

  def test_wall_clock_save_and_restore_checkpoint(self):
    ckpt_dir = epath.Path(self.create_tempdir().full_path) / 'wc_ckpts'
    mgr = ocp.CheckpointManager(
        directory=ckpt_dir,
        checkpointers={
            'train_state': ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
            ),
        },
    )
    model, params, x = _init(remat=False, seed=99)
    state = train.utils.TrainState(
        step=1357,
        params=params,
        ema_params=params,
        batch_stats={},
        opt_state={'count': 1357, 'opt': jax.tree.map(jnp.zeros_like, params)},
        min_s_mmd_train=0.25,
    )
    should_save, is_wc = train.should_save_checkpoint(
        step=1357,
        save_checkpoint_steps=10000,
        last_checkpoint_time=0.0,
        save_checkpoint_secs=900,
        now=905.0,
        max_steps=100000,
    )
    self.assertTrue(should_save)
    self.assertTrue(is_wc)

    mgr.save(1357, items={'train_state': state})
    mgr.wait_until_finished()
    self.assertEqual(mgr.latest_step(), 1357)

    restored = mgr.restore(1357, items={'train_state': state})['train_state']
    self.assertEqual(restored.step, 1357)
    self.assertEqual(restored.opt_state['count'], 1357)
    np.testing.assert_allclose(
        model.apply({'params': restored.params}, x),
        model.apply({'params': params}, x),
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()


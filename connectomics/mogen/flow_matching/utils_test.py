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

from absl.testing import absltest
from connectomics.mogen.flow_matching import utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import synjax


class UtilsTest(absltest.TestCase):

  def test_simple_embs(self):
    pc = jax.random.uniform(jax.random.PRNGKey(0), (8, 256, 3))
    embs = utils.simple_embs(pc, mst=True)
    self.assertEqual(embs.shape, (8, 10))

  def test_mst(self):
    n_nodes = 128
    adj = jax.random.uniform(
        jax.random.PRNGKey(0), (2, n_nodes, n_nodes), minval=0.1, maxval=1.0
    )
    adj = (adj + adj.transpose((0, 2, 1))) / 2
    adj = adj.at[..., jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0)

    mst_adj_prim = utils.prim_mst(adj)
    mst_adj_synjax = synjax.SpanningTreeCRF(
        -adj, directed=False, projective=False, single_root_edge=False
    ).argmax()  # slow
    self.assertTrue(jnp.allclose(mst_adj_prim, mst_adj_synjax))

  def test_moment_embedding(self):
    pc = jax.random.uniform(jax.random.PRNGKey(0), (8, 256, 3))
    embs = utils.moment_embedding(pc, n=4)
    self.assertEqual(embs.shape, (8, 15))

  def test_prep_data_combine_samples_pointinfinity(
      self,
  ):
    batch = {
        'coord': jnp.array(np.random.rand(4, 10, 3), dtype=jnp.float32),
        'feat': jnp.array(np.random.rand(4, 10, 2), dtype=jnp.float32),
        '_dataset_index': jnp.array(
            np.arange(4).reshape(4, 1), dtype=jnp.int32
        ),
    }

    config = ml_collections.ConfigDict()
    config.model_type = 'pointinfinity'
    config.use_feat = True
    config = ml_collections.FrozenConfigDict(config)

    n_combine_samples = 2
    n_points = 5
    coord, feat, cond = utils.prep_data(
        batch,
        n_combine_samples=n_combine_samples,
        n_points=n_points,
        coord_scale=1.0,
        feat_scale=1.0,
        cond_mode='moment',
        use_feat=config.use_feat,
    )

    self.assertEqual(coord.shape, (2, 10, 3))
    self.assertEqual(feat.shape, (2, 10, 2))
    self.assertEqual(cond.shape, (2, 15))

  def test_plot_point_clouds_with_combine_mask(
      self,
  ):
    pc = jax.random.uniform(jax.random.PRNGKey(0), (2, 10, 3))
    fig = utils.plot_point_clouds(pc, n_combine_samples=2)
    self.assertIsNotNone(fig)


class _TinyVelocity(utils.nn.Module):
  """Tiny velocity model with the call signature used by guided_apply."""

  @utils.nn.compact
  def __call__(
      self,
      coord,
      feat=None,
      t=None,
      cond=None,
      point_cond_mask=None,
      deterministic=True,
  ):
    del feat, cond, point_cond_mask, deterministic
    h = utils.nn.Dense(8)(coord) + utils.nn.Dense(8)(t[:, None, None])
    return utils.nn.Dense(3)(jnp.tanh(h))


def _tiny_setup():
  model = _TinyVelocity()
  x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 3))
  t = jnp.ones((4,))
  params = model.init(jax.random.PRNGKey(1), coord=x, t=t)['params']
  params_bad = model.init(jax.random.PRNGKey(2), coord=x, t=t)['params']
  state = utils.TrainState(
      step=0, params=params, ema_params=params, batch_stats={}, opt_state=None
  )
  return model, state, params_bad


def _legacy_generate_samples(
    model, state, n_samples, sample_shape, rng, n_steps, schedule
):
  """The pre-change default sampler: explicit midpoint on ema_params."""
  x_0 = jax.random.normal(rng, (n_samples, *sample_shape))
  timesteps = utils.schedules.t_schedule(
      jnp.linspace(0.0, 1.0, n_steps + 1), schedule
  )

  def body_fun(i, x_t):
    return utils.generate_step_midpoint(
        model,
        state,
        x_t,
        t_start=timesteps[i],
        t_end=timesteps[i + 1],
        cond=None,
        point_cond_mask=None,
        guidance_scale=None,
        guide=False,
    )

  return jax.lax.fori_loop(0, n_steps, body_fun, x_0)


_GEN_KWARGS = dict(
    n_samples=4, sample_shape=(16, 3), n_steps=6, schedule='cosine_2.0'
)


class SamplerTest(absltest.TestCase):

  def test_default_path_bit_identical(self):
    model, state, _ = _tiny_setup()
    rng = jax.random.PRNGKey(3)
    expected = _legacy_generate_samples(model, state, rng=rng, **_GEN_KWARGS)
    np.testing.assert_array_equal(
        utils.generate_samples(model, state, rng=rng, **_GEN_KWARGS), expected
    )
    np.testing.assert_array_equal(
        utils.generate_samples(
            model,
            state,
            rng=rng,
            solver='midpoint',
            autoguidance_params=None,
            autoguidance_weight=1.0,
            **_GEN_KWARGS,
        ),
        expected,
    )

  def test_generic_steps_match_legacy_steps(self):
    model, state, _ = _tiny_setup()
    x = jax.random.normal(jax.random.PRNGKey(4), (4, 16, 3))
    t0, t1 = jnp.array([0.2]), jnp.array([0.35])
    v = utils.make_velocity_fn(model, state)
    np.testing.assert_allclose(
        utils.ode_step(v, x, t0, t1, 'midpoint'),
        utils.generate_step_midpoint(model, state, x, t0, t1),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        utils.ode_step(v, x, t0, t1, 'rk4'),
        utils.generate_step_rk4(model, state, x, t0, t1),
        rtol=0,
        atol=1e-6,
    )

  def test_autoguidance_weight_one_equals_good_model(self):
    model, state, params_bad = _tiny_setup()
    rng = jax.random.PRNGKey(5)
    good = utils.generate_samples(model, state, rng=rng, **_GEN_KWARGS)
    ag_w1 = utils.generate_samples(
        model,
        state,
        rng=rng,
        autoguidance_params=params_bad,
        autoguidance_weight=1.0,
        **_GEN_KWARGS,
    )
    np.testing.assert_allclose(ag_w1, good, rtol=0, atol=1e-5)
    ag_same_model = utils.generate_samples(
        model,
        state,
        rng=rng,
        autoguidance_params=state.ema_params,
        autoguidance_weight=1.5,
        **_GEN_KWARGS,
    )
    np.testing.assert_allclose(ag_same_model, good, rtol=0, atol=1e-5)
    ag_w15 = utils.generate_samples(
        model,
        state,
        rng=rng,
        autoguidance_params=params_bad,
        autoguidance_weight=1.5,
        **_GEN_KWARGS,
    )
    self.assertGreater(float(jnp.abs(ag_w15 - good).max()), 1e-3)

  def test_heun_exact_on_linear_field(self):
    a = jax.random.normal(jax.random.PRNGKey(6), (2, 5, 3))
    b = jax.random.normal(jax.random.PRNGKey(7), (2, 5, 3))
    x0 = jax.random.normal(jax.random.PRNGKey(8), (2, 5, 3))
    velocity = lambda x, t: a + b * t[:, None, None]
    t0, t1 = 0.2, 0.7
    exact = x0 + a * (t1 - t0) + b * (t1**2 - t0**2) / 2
    t0_arr, t1_arr = jnp.array([t0]), jnp.array([t1])
    heun = utils.ode_step(velocity, x0, t0_arr, t1_arr, 'heun')
    np.testing.assert_allclose(heun, exact, rtol=0, atol=1e-5)
    euler = utils.ode_step(velocity, x0, t0_arr, t1_arr, 'euler')
    self.assertGreater(float(jnp.abs(euler - exact).max()), 1e-3)

  def test_ab2_exact_on_linear_field(self):
    a = jax.random.normal(jax.random.PRNGKey(6), (2, 5, 3))
    b = jax.random.normal(jax.random.PRNGKey(7), (2, 5, 3))
    x0 = jax.random.normal(jax.random.PRNGKey(8), (2, 5, 3))
    velocity = lambda x, t: a + b * t[:, None, None]
    t0, t1, t2 = 0.1, 0.35, 0.8
    t0_arr, t1_arr, t2_arr = jnp.array([t0]), jnp.array([t1]), jnp.array([t2])
    x1, k0 = utils.ode_step(velocity, x0, t0_arr, t1_arr, 'ab2', return_k=True)
    x2_ab2, _ = utils.ode_step(
        velocity,
        x1,
        t1_arr,
        t2_arr,
        'ab2',
        k_prev=k0,
        dt_prev=t1_arr - t0_arr,
        return_k=True,
    )
    exact = x1 + a * (t2 - t1) + b * (t2**2 - t1**2) / 2
    np.testing.assert_allclose(x2_ab2, exact, rtol=0, atol=1e-5)

  def test_solver_order_and_exactness(self):
    # dx/dt = v(x, t) = -x. Exact solution: x(t) = x(0) * exp(-t)
    def velocity(x, t):
      return -x

    x0 = jnp.array([[[1.0]]])
    t_start = jnp.array([0.0])
    t_end = jnp.array([1.0])

    # We test order by checking error scaling as dt halves
    for solver, expected_order in (
        ('euler', 1),
        ('midpoint', 2),
        ('heun', 2),
        ('ab2', 2),
        ('rk4', 4),
    ):
      errors = []
      for steps in (10, 20):
        if solver == 'rk4':
          steps //= 2

        # We manually simulate the loop or use ode_step
        # To avoid setting up full config, we just run ode_step in a loop
        dt = 1.0 / steps
        x_t = x0
        k_prev = jnp.zeros_like(x0)
        dt_prev = jnp.array([dt])
        for i in range(steps):
          t_i = jnp.array([i * dt])
          t_next = jnp.array([(i + 1) * dt])
          x_next, k_next = utils.ode_step(
              velocity,
              x_t,
              t_i,
              t_next,
              solver=solver,
              k_prev=k_prev,
              dt_prev=dt_prev,
              return_k=True,
          )
          x_t = x_next
          k_prev = k_next
          dt_prev = t_next - t_i

        errors.append(float(jnp.abs(x_t - jnp.exp(-1.0)).max()))

      error_ratio = errors[0] / errors[1]

      # For a p-th order method, halving step size reduces error by ~2^p
      # Expected ratio: ~ 2^expected_order
      self.assertGreater(
          error_ratio,
          2 ** (expected_order - 0.5),
          msg=f'{solver} order check failed',
      )

  def test_solver_nfe_and_generic_samplers_run(self):
    calls = []

    def velocity(x, t):
      del t
      calls.append(1)
      return jnp.zeros_like(x)

    x0 = jnp.zeros((1, 2, 3))
    for solver, nfe in (
        ('euler', 1),
        ('midpoint', 2),
        ('heun', 2),
        ('ab2', 1),
        ('rk4', 4),
    ):
      calls.clear()
      utils.ode_step(velocity, x0, jnp.array([0.0]), jnp.array([0.5]), solver)
      self.assertLen(calls, nfe, msg=solver)
    with self.assertRaises(ValueError):
      utils.ode_step(velocity, x0, jnp.array([0.0]), jnp.array([0.5]), 'foo')
    model, state, _ = _tiny_setup()
    for solver in utils.SOLVERS:
      out = utils.generate_samples(
          model, state, rng=jax.random.PRNGKey(9), solver=solver, **_GEN_KWARGS
      )
      self.assertEqual(out.shape, (4, 16, 3))
      self.assertTrue(bool(jnp.all(jnp.isfinite(out))))


class GuidanceIntervalTest(absltest.TestCase):

  def test_velocity_interval(self):
    model, state, params_bad = _tiny_setup()
    x = jax.random.normal(jax.random.PRNGKey(10), (4, 16, 3))
    good = utils.make_velocity_fn(model, state)
    full = utils.make_velocity_fn(
        model, state, autoguidance_params=params_bad, autoguidance_weight=1.5
    )
    limited = utils.make_velocity_fn(
        model,
        state,
        autoguidance_params=params_bad,
        autoguidance_weight=1.5,
        autoguidance_t_min=0.3,
        autoguidance_t_max=0.7,
    )
    explicit_full = utils.make_velocity_fn(
        model,
        state,
        autoguidance_params=params_bad,
        autoguidance_weight=1.5,
        autoguidance_t_min=0.0,
        autoguidance_t_max=1.0,
    )
    for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
      t = jnp.full((4,), t_val)
      np.testing.assert_array_equal(explicit_full(x, t), full(x, t))
      expected = full(x, t) if 0.3 <= t_val <= 0.7 else good(x, t)
      np.testing.assert_allclose(limited(x, t), expected, rtol=0, atol=1e-6)
    # Guidance actually changes the velocity inside the interval.
    t = jnp.full((4,), 0.5)
    self.assertGreater(float(jnp.abs(full(x, t) - good(x, t)).max()), 1e-4)

  def test_generate_samples_interval(self):
    model, state, params_bad = _tiny_setup()
    rng = jax.random.PRNGKey(11)
    kw = dict(autoguidance_params=params_bad, autoguidance_weight=1.5)
    full = utils.generate_samples(model, state, rng=rng, **kw, **_GEN_KWARGS)
    np.testing.assert_array_equal(
        utils.generate_samples(
            model,
            state,
            rng=rng,
            autoguidance_t_min=0.0,
            autoguidance_t_max=1.0,
            **kw,
            **_GEN_KWARGS,
        ),
        full,
    )
    # An empty interval gives the unguided (good-model) samples.
    empty = utils.generate_samples(
        model,
        state,
        rng=rng,
        autoguidance_t_min=2.0,
        autoguidance_t_max=3.0,
        **kw,
        **_GEN_KWARGS,
    )
    good = utils.generate_samples(model, state, rng=rng, **_GEN_KWARGS)
    np.testing.assert_allclose(empty, good, rtol=0, atol=1e-5)
    partial = utils.generate_samples(
        model,
        state,
        rng=rng,
        autoguidance_t_min=0.5,
        **kw,
        **_GEN_KWARGS,
    )
    self.assertGreater(float(jnp.abs(partial - full).max()), 1e-5)
    self.assertGreater(float(jnp.abs(partial - good).max()), 1e-5)

  def test_noise_temp(self):
    model, state, _ = _tiny_setup()
    rng = jax.random.PRNGKey(12)
    default = utils.generate_samples(model, state, rng=rng, **_GEN_KWARGS)
    np.testing.assert_array_equal(
        utils.generate_samples(
            model, state, rng=rng, noise_temp=1.0, **_GEN_KWARGS
        ),
        default,
    )
    noise = jax.random.normal(
        rng, (_GEN_KWARGS['n_samples'], *_GEN_KWARGS['sample_shape'])
    )
    cold = utils.generate_samples(
        model, state, rng=rng, noise_temp=0.5, **_GEN_KWARGS
    )
    explicit = utils.generate_samples(
        model, state, rng=None, noise=0.5 * noise, **_GEN_KWARGS
    )
    np.testing.assert_allclose(cold, explicit, rtol=0, atol=1e-6)
    self.assertGreater(float(jnp.abs(cold - default).max()), 1e-3)


class UpdateScaleTest(absltest.TestCase):

  def _setup(self):
    model = _TinyVelocity()
    coord = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 3))
    params = model.init(jax.random.PRNGKey(1), coord=coord, t=jnp.ones((2,)))[
        'params'
    ]
    optimizer = utils.optax.sgd(0.1)
    state = utils.TrainState(
        step=0,
        params=params,
        ema_params=params,
        batch_stats={},
        opt_state=optimizer.init(params),
    )
    return model, state, optimizer, coord

  def _step(self, update_scale):
    model, state, optimizer, coord = self._setup()
    kw = {} if update_scale == 'absent' else {'update_scale': update_scale}
    new_state, _ = utils.update_state(
        model=model,
        state=state,
        optimizer=optimizer,
        coord=coord,
        feat=None,
        rng=jax.random.PRNGKey(2),
        polyak_decay=0.0,
        **kw,
    )
    delta = jax.tree_util.tree_map(
        lambda a, b: a - b, new_state.params, state.params
    )
    return new_state, delta

  def test_update_scale(self):
    _, d_absent = self._step('absent')
    _, d_none = self._step(None)
    _, d_one = self._step(1.0)
    _, d_half = self._step(jnp.float32(0.5))
    s_zero, d_zero = self._step(0.0)
    leaves = jax.tree_util.tree_leaves
    for a, b, c in zip(leaves(d_absent), leaves(d_none), leaves(d_one)):
      np.testing.assert_array_equal(a, b)
      np.testing.assert_allclose(a, c, rtol=1e-6, atol=1e-8)
    for a, h in zip(leaves(d_absent), leaves(d_half)):
      # Deltas are (params + update) - params in float32, so they carry an
      # absolute rounding error of ~1 ulp of the O(1) params (~6e-8).
      np.testing.assert_allclose(h, 0.5 * a, rtol=1e-5, atol=3e-7)
    for z in leaves(d_zero):
      np.testing.assert_array_equal(z, np.zeros_like(z))
    self.assertEqual(int(s_zero.step), 1)
    self.assertGreater(
        max(float(jnp.abs(a).max()) for a in leaves(d_absent)), 0.0
    )

  def test_update_scale_prodigy(self):
    model = _TinyVelocity()
    coord = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 3))
    params = model.init(jax.random.PRNGKey(1), coord=coord, t=jnp.ones((2,)))[
        'params'
    ]
    optimizer = utils.optax.contrib.prodigy(learning_rate=0.5)

    def _step_prodigy(scale):
      state = utils.TrainState(
          step=100,
          params=params,
          ema_params=params,
          batch_stats={},
          opt_state=optimizer.init(params),
      )
      new_state, _ = utils.update_state(
          model=model,
          state=state,
          optimizer=optimizer,
          coord=coord,
          feat=None,
          rng=jax.random.PRNGKey(2),
          polyak_decay=0.0,
          update_scale=scale,
      )
      delta = jax.tree_util.tree_map(
          lambda a, b: a - b, new_state.params, state.params
      )
      return new_state, delta

    s_full, d_full = _step_prodigy(
        utils.schedules.anneal_factor(100, begin=100, steps=1000, final=0.0)
    )
    s_half, d_half = _step_prodigy(
        utils.schedules.anneal_factor(600, begin=100, steps=1000, final=0.0)
    )
    s_zero, d_zero = _step_prodigy(
        utils.schedules.anneal_factor(1100, begin=100, steps=1000, final=0.0)
    )

    leaves = jax.tree_util.tree_leaves
    self.assertGreater(
        max(float(jnp.abs(a).max()) for a in leaves(d_full)), 0.0
    )
    for a, h in zip(leaves(d_full), leaves(d_half)):
      np.testing.assert_allclose(h, 0.5 * a, rtol=1e-5, atol=3e-7)
    for z in leaves(d_zero):
      np.testing.assert_array_equal(z, np.zeros_like(z))
    self.assertEqual(int(s_full.step), 101)
    self.assertEqual(int(s_half.step), 101)
    self.assertEqual(int(s_zero.step), 101)
    self.assertIsNotNone(s_full.opt_state)
    self.assertIsNotNone(s_zero.opt_state)


class SchedulesTest(absltest.TestCase):

  def test_logit_normal(self):
    u = jnp.linspace(0.0, 1.0, 101)
    t = utils.schedules.t_schedule(u, 'logitnormal_0.0_1.0')
    self.assertTrue(bool(jnp.all(jnp.isfinite(t))))
    self.assertAlmostEqual(float(t[0]), 0.0, places=6)
    self.assertAlmostEqual(float(t[-1]), 1.0, places=6)
    self.assertAlmostEqual(float(t[50]), 0.5, places=6)
    self.assertTrue(bool(jnp.all(jnp.diff(t) >= 0)))
    shifted = utils.schedules.t_schedule(u, 'logitnormal_1.0_0.5')
    self.assertAlmostEqual(float(shifted[50]), float(jax.nn.sigmoid(1.0)), 5)
    # Mid-heavy: more mass in [0.25, 0.75] than uniform sampling.
    samples = utils.schedules.t_schedule(
        jax.random.uniform(jax.random.PRNGKey(0), (20000,)),
        'logitnormal_0.0_1.0',
    )
    frac_mid = float(jnp.mean((samples > 0.25) & (samples < 0.75)))
    self.assertGreater(frac_mid, 0.65)
    self.assertLess(frac_mid, 0.75)

  def test_anneal_factor(self):
    f = utils.schedules.anneal_factor
    self.assertAlmostEqual(float(f(0, begin=10, steps=100)), 1.0)
    self.assertAlmostEqual(float(f(10, begin=10, steps=100)), 1.0)
    self.assertAlmostEqual(float(f(60, begin=10, steps=100)), 0.5)
    self.assertAlmostEqual(float(f(110, begin=10, steps=100)), 0.0)
    self.assertAlmostEqual(float(f(10**6, begin=10, steps=100)), 0.0)
    self.assertAlmostEqual(
        float(f(60, begin=10, steps=100, final=0.2)), 0.6, places=6
    )
    self.assertAlmostEqual(
        float(f(35, begin=10, steps=100, shape='cosine')),
        1.0 - 0.5 * (1.0 - np.cos(np.pi * 0.25)),
        places=6,
    )
    with self.assertRaises(ValueError):
      f(0, begin=0, steps=0)
    with self.assertRaises(ValueError):
      f(0, begin=0, steps=10, shape='foo')


class DiversityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    self.real = rng.normal(size=(3000, 10)).astype(np.float32)
    self.gen = rng.normal(size=(512, 10)).astype(np.float32)

  def test_matched_distribution(self):
    d = utils.diversity_metrics(self.gen, self.real, k=5, n_real=2048)
    self.assertAlmostEqual(d['std_ratio'], 1.0, delta=0.08)
    self.assertGreater(d['std_ratio_min'], 0.85)
    # Expected coverage ~ 1 - (1 - 5/2048)^512 ~ 0.71 for matched samples.
    self.assertGreater(d['coverage'], 0.55)
    self.assertLess(d['coverage'], 0.85)
    self.assertGreater(d['precision'], 0.8)
    self.assertGreater(d['recall'], 0.5)

  def test_detects_shrinkage_and_collapse(self):
    matched = utils.diversity_metrics(self.gen, self.real)
    half = utils.diversity_metrics(0.5 * self.gen, self.real)
    collapsed = utils.diversity_metrics(
        np.zeros_like(self.gen) + 1e-3 * self.gen, self.real
    )
    self.assertAlmostEqual(half['std_ratio'], 0.5, delta=0.05)
    self.assertLess(half['coverage'], matched['coverage'] - 0.05)
    self.assertLess(half['recall'], matched['recall'])
    self.assertLess(collapsed['std_ratio'], 0.01)
    self.assertLess(collapsed['coverage'], 0.05)
    self.assertLess(collapsed['recall'], 0.05)

  def test_reference_baselines_ordering(self):
    ref = utils.reference_diversity_baselines(self.real, n_gen=512)
    self.assertLess(
        ref['ref_real_s_mmd_val_mst'], ref['ref_shrink50_s_mmd_val_mst']
    )
    self.assertLess(
        ref['ref_shrink50_s_mmd_val_mst'], ref['ref_collapsed_s_mmd_val_mst']
    )
    self.assertGreater(ref['ref_real_coverage'], ref['ref_shrink50_coverage'])
    self.assertAlmostEqual(ref['ref_shrink50_std_ratio'], 0.5, delta=0.06)
    self.assertLess(ref['ref_collapsed_std_ratio'], 1e-3)


if __name__ == '__main__':
  absltest.main()

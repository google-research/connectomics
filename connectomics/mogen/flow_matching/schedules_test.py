# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
from connectomics.mogen.flow_matching import schedules
import jax
import jax.numpy as jnp


class SchedulesTest(absltest.TestCase):

  def test_logitnormal_distribution(self):
    u = jnp.linspace(1e-5, 1 - 1e-5, 100000)

    t1 = schedules.t_schedule(u, 'logitnormal_0.0_1.0')
    self.assertAlmostEqual(jnp.mean(t1).item(), 0.5, places=2)
    self.assertAlmostEqual(jnp.median(t1).item(), 0.5, places=2)

    t2 = schedules.t_schedule(u, 'logitnormal_-0.5_1.0')
    expected_median = jax.nn.sigmoid(-0.5).item()
    self.assertAlmostEqual(jnp.median(t2).item(), expected_median, places=2)

    self.assertTrue(jnp.all(t1 > 0))
    self.assertTrue(jnp.all(t1 < 1))

  def test_edm_schedule(self):
    u = jnp.linspace(0.0, 1.0, 1000)
    t = schedules.t_schedule(u, 'edm_7.0')
    expected = 1.0 - (1.0 - u) ** 7.0
    self.assertTrue(jnp.allclose(t, expected))
    t_rho7 = schedules.t_schedule(u, 'rho7')
    self.assertTrue(jnp.allclose(t_rho7, expected))

  def test_linear_schedule(self):
    u = jnp.linspace(0, 1, 1000)
    t = schedules.t_schedule(u, 'linear')
    self.assertTrue(jnp.allclose(t, u))
    t_uniform = schedules.t_schedule(u, 'uniform')
    self.assertTrue(jnp.allclose(t_uniform, u))

  def test_cosine_schedule(self):
    u = jnp.linspace(0.0, 1.0, 1000)
    t_cos = schedules.t_schedule(u, 'cosine')
    expected_cos = 0.5 * (1.0 - jnp.cos(u * jnp.pi))
    self.assertTrue(jnp.allclose(t_cos, expected_cos, atol=1e-5))


if __name__ == '__main__':
  absltest.main()

  def test_adamc_decoupled_weight_decay(self):
    import optax
    # Simulate config values
    peak_lr = 1.0
    wd = 0.1

    # Test 1: constant schedule = peak => AdamC == AdamW
    adamc_const = optax.adamw(
        learning_rate=peak_lr, weight_decay=lambda c: wd * peak_lr / peak_lr
    )
    adamw_const = optax.adamw(learning_rate=peak_lr, weight_decay=wd)

    params = {'w': jnp.array([1.0])}
    grads = {'w': jnp.array([0.0])}  # 0 gradient to isolate weight decay

    state_c = adamc_const.init(params)
    updates_c, _ = adamc_const.update(grads, state_c, params)

    state_w = adamw_const.init(params)
    updates_w, _ = adamw_const.update(grads, state_w, params)

    self.assertTrue(jnp.allclose(updates_c['w'], updates_w['w']))

    # Test 2: sched = 0.5 * peak => decay term == 0.25 * wd * peak * param
    # optax.adamw applies: param_update = -lr * (grad + wd_coeff * param)
    # For AdamC, wd_coeff = wd * (lr / peak)
    # So update = -lr * wd * (lr / peak) * param
    # If lr = 0.5 * peak: update = - (0.5 * peak) * wd * (0.5 * peak / peak)
    # * param = -0.25 * peak * wd * param

    lr_half = 0.5 * peak_lr
    adamc_half = optax.adamw(
        learning_rate=lr_half, weight_decay=lambda c: wd * lr_half / peak_lr
    )
    state_half = adamc_half.init(params)
    updates_half, _ = adamc_half.update(grads, state_half, params)

    # Expected decay: -0.25 * wd * peak_lr * param
    expected_decay = -0.25 * wd * peak_lr * params['w']
    self.assertTrue(jnp.allclose(updates_half['w'], expected_decay))

  def test_find_sf_state(self):
    import optax

    # Test finding SF eval params in a chain with clip_by_global_norm
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.contrib.schedule_free_adamw(learning_rate=0.1),
    )

    params = {'w': jnp.array([1.0])}
    opt_state = optimizer.init(params)

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

    eval_params = find_sf_state(opt_state, params)
    self.assertIsNotNone(eval_params)
    self.assertIn('w', eval_params)

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
"""Feature regression training."""

import collections
import time
from typing import Any

from absl import logging
from clu import metric_writers
from connectomics.mogen.flow_matching import utils
from etils import epath
from ffn.inference import storage
import flax
import flax.linen as nn
import grain.tensorflow as grain
import jax
from jax import sharding
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
  feat = rng.standard_normal((batch_size, n_points, n_feat)).astype(np.float32)
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


def compute_feature_regression_loss(
    model: nn.Module,
    variables: flax.core.FrozenDict[str, Any] | dict[str, Any],
    rng: jax.Array,
    coord: jax.Array,
    feat: jax.Array,
    is_train: bool,
    clip_last_feat_abs_val: float = 0.0,
    **unused_kwargs,
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
    clip_last_feat_abs_val: If > 0, clip absolute value of last feature to this
      value.
    **unused_kwargs: Additional keyword arguments.

  Returns:
    Loss value and auxiliary outputs.
  """
  del unused_kwargs

  apply_out = model.apply(
      variables,
      coord=coord,
      rngs={'dropout': rng},
      mutable=is_train,
      deterministic=not is_train,
  )
  if is_train:
    pred_dx_t, mutable = apply_out
    batch_stats = mutable.get('batch_stats', None)
  else:
    pred_dx_t = apply_out
    batch_stats = None

  if clip_last_feat_abs_val > 0:
    # this is the curvature feature with potential outliers
    feat = feat.at[:, :, -1].set(
        jnp.clip(
            feat[:, :, -1], -clip_last_feat_abs_val, clip_last_feat_abs_val
        )
    )

  loss = jnp.mean(jnp.square(pred_dx_t - feat))

  norm_pred = jnp.linalg.norm(pred_dx_t[:, :, :3], axis=-1)
  norm_feat = jnp.linalg.norm(feat[:, :, :3], axis=-1)
  cosine_sim = jnp.mean(
      jnp.sum(pred_dx_t[:, :, :3] * feat[:, :, :3], axis=-1)
      # :3 for the vector feature
      / (norm_pred * norm_feat)
  )

  last_mse = jnp.mean(jnp.square(pred_dx_t[:, :, 3:] - feat[:, :, 3:]))

  pred_norm_mean = jnp.mean(norm_pred)
  pred_norm_std = jnp.std(norm_pred)

  return loss, {
      'loss': loss,
      'batch_stats': batch_stats,
      'cosine_sim': cosine_sim,
      'last_mse': last_mse,
      'pred_norm_mean': pred_norm_mean.mean(),
      'pred_norm_std': pred_norm_std.mean(),
      'norm_pred': norm_pred.mean(),
      'norm_feat': norm_feat.mean(),
      'feat_last_std': feat[:, :, -1].std(),
  }


def train_feature_regression(
    config: ml_collections.ConfigDict, log_dir: str
) -> jax.Array:
  """Trains a feature regression model.

  Args:
    config: Configuration for training.
    log_dir: Directory to write logs.

  Returns:
    Generated samples after training.
  """
  workdir = epath.Path(config.workdir) / config.name_str
  log_dir = epath.Path(log_dir) / config.name_str
  logging.info('workdir: %s, log_dir: %s', workdir, log_dir)

  workdir.mkdir(parents=True, exist_ok=True)
  writer = metric_writers.create_default_writer(
      log_dir, just_logging=jax.process_index() > 0
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
  logging.debug('config.n_combine_samples = %d', config.n_combine_samples)
  logging.debug(
      'Effective batch_size for get_dataloaders = %d',
      config.batch_size * config.n_combine_samples,
  )
  train_iter = iter(train_dataloader)

  first_batch = next(train_iter)
  logging.debug("first_batch['coord'].shape = %s", first_batch['coord'].shape)
  init_coord, init_feat, _ = utils.prep_data(
      first_batch,
      config.coord_scale,
      config.feat_scale,
      config.n_points // config.n_combine_samples,
      n_combine_samples=config.n_combine_samples,
      use_feat=True,
      dst_index_cond=config.train_set == 'mixed',
      # TODO(riegerfr): make dst_index_cond cleaner
      # (i.e. no dependency on train set string here)
  )
  logging.info('init_feat.std = %s', init_feat.std(axis=(0, 1)))
  logging.debug('init_coord.shape = %s', init_coord.shape)
  logging.debug('config.batch_size for assertion = %d', config.batch_size)
  batch_dim, n_points_dim, coord_dim = init_coord.shape
  assert n_points_dim == config.n_points
  assert coord_dim == 3
  assert batch_dim // config.num_devices == config.batch_size, (
      f'n_train_samples was {n_train_samples} but batch_dim is'
      f' {batch_dim} and config.batch_size is {config.batch_size} '
      f'and config.num_devices is {config.num_devices}'
  )

  with storage.atomic_file(str(workdir / 'init_data.npz'), 'wb') as f:
    np.savez_compressed(
        f, coord=np.asarray(init_coord), feat=np.asarray(init_feat)
    )

  feat_shape = init_feat.shape[2]
  # Check that coordinates are scaled to [-1, 1]; allowing for a small
  # fraction of outliers.
  assert (
      (-1 < init_coord) & (init_coord < 1)
  ).mean() > 0.9, 'Coordinates should be in [-1, 1] after scaling.'

  config.out_dim = feat_shape

  model, params, batch_stats = utils.get_model(
      model_rng,
      init_coord=init_coord[:1],
      init_feat=None,
      config=config,
      n_combine_samples=config.n_combine_samples,
  )
  optimizer = utils.get_optimizer(config)(
      learning_rate=config.lr, weight_decay=config.wd
  )
  if config.clip_grad > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_grad), optimizer
    )

  state = utils.TrainState(
      step=0,
      params=params,
      ema_params=params,
      batch_stats=batch_stats,
      opt_state=optimizer.init(params),
  )

  with storage.atomic_file(str(workdir / 'init_coord.html'), 'w') as f:
    utils.plot_point_clouds(
        init_coord[:16], n_combine_samples=config.n_combine_samples
    ).write_html(f)

  latest_checkpoint_manager = ocp.CheckpointManager(
      directory=workdir / 'checkpoints',
      checkpointers={
          'train_state': ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
          'train_iter': ocp.Checkpointer(grain.OrbaxCheckpointHandler()),  # pytype:disable=wrong-arg-types
      },
      options=ocp.CheckpointManagerOptions(max_to_keep=3),
  )  # to restore after preemption
  best_checkpoint_manager = ocp.CheckpointManager(
      directory=workdir / 'best_checkpoints',
      checkpointers={
          'train_state': ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
      },
      options=ocp.CheckpointManagerOptions(max_to_keep=3),
  )  # for inference, workaround: https://github.com/google/orbax/issues/526

  latest_step = latest_checkpoint_manager.latest_step()
  restored_data = None
  if latest_step is not None:
    try:
      restored_data = latest_checkpoint_manager.restore(
          latest_step,
          items={'train_state': state, 'train_iter': train_iter},
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('Could not restore checkpoint: %s', e)

  if restored_data:
    state = restored_data['train_state']
    train_iter = restored_data['train_iter']
    logging.info(
        'Restored checkpoint for step %d',
        latest_step,
    )
  else:
    logging.info('Starting training from scratch.')
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

  mesh = sharding.Mesh(np.array(jax.devices()), ('batch',))
  batch_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec('batch'))
  replicate_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec())
  logging.info('Device mesh: %r', mesh)

  state = jax.device_put(state, replicate_sharding)

  train_start_time = time.time()
  running_train_step = 0
  running_train_loss = 0.0
  running_train_loss_squared = 0.0  # For variance calculation
  log_dict = {
      'sum': collections.defaultdict(float),
      'count': collections.defaultdict(int),
  }

  def train_step(state, batch):
    """Performs a single training step."""
    update_rng = jax.random.fold_in(train_rng, state.step)

    coord, feat, _ = utils.prep_data(
        batch,
        config.coord_scale,
        config.feat_scale,
        config.n_points // config.n_combine_samples,
        n_combine_samples=config.n_combine_samples,
        use_feat=True,
        dst_index_cond=config.train_set == 'mixed',
    )

    return utils.update_state(
        model=model,
        state=state,
        optimizer=optimizer,
        coord=coord,
        feat=feat,
        rng=update_rng,
        polyak_decay=config.polyak_decay,
        comp_loss=lambda *args, **kwargs: compute_feature_regression_loss(
            *args,
            **kwargs,
            clip_last_feat_abs_val=config.clip_last_feat_abs_val,
        ),
    )

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
  )

  def val_step(state, batch, val_rng):
    """Computes the validation loss."""

    coord, feat, _ = utils.prep_data(
        batch,
        config.coord_scale,
        config.feat_scale,
        config.n_points // config.n_combine_samples,
        n_combine_samples=config.n_combine_samples,
        use_feat=True,
        dst_index_cond=config.train_set == 'mixed',
    )

    variables = {'params': state.ema_params}
    if state.batch_stats:
      variables['batch_stats'] = state.batch_stats
    return compute_feature_regression_loss(
        model,
        variables,
        val_rng,
        coord=coord,
        feat=feat,
        is_train=False,
        clip_last_feat_abs_val=config.clip_last_feat_abs_val,
    )

  p_val_step = jax.jit(
      val_step,
      in_shardings=(
          replicate_sharding,  # state
          batch_sharding,  # data
          replicate_sharding,  # rng
      ),
      out_shardings=(
          replicate_sharding,  # loss
          replicate_sharding,  # aux
      ),
  )

  while state.step < config.max_steps:
    batch = next(train_iter)

    batch = {
        'coord': batch['coord'],
        'feat': batch['feat'],
        '_dataset_index': batch.get('_dataset_index'),
    }
    assert (
        (-1 < batch['coord']) & (batch['coord'] < 1)
    ).mean() > 0.9, 'Coordinates should be in [-1, 1] after scaling.'

    state, aux = p_train_step(state, batch)
    for k, v in aux.items():
      if v is not None:
        log_dict['sum'][k] += v
        log_dict['count'][k] += 1

    running_train_loss += aux['loss']
    running_train_loss_squared += aux['loss'] ** 2
    running_train_step += 1

    if (
        state.step % config.log_train_every_steps == 0
        or state.step == config.max_steps
        or state.step == 10
    ):
      train_loss = running_train_loss / running_train_step
      train_loss_variance = (
          running_train_loss_squared / running_train_step - train_loss**2
      )
      logging.info(
          'Step: %d, Train Loss: %.5f, Train Loss Variance: %.5f, Time: %.2fs',
          state.step,
          train_loss,
          train_loss_variance,
          time.time() - train_start_time,
      )
      writer.write_scalars(
          int(state.step),
          {
              'train_loss': float(train_loss),
              'train_loss_variance': float(train_loss_variance),
          },
      )

      log_scalars = {}
      for k, v in log_dict['sum'].items():
        if log_dict['count'][k] > 0:
          log_scalars[f'train_{k}'] = v / log_dict['count'][k]
        else:
          log_scalars[f'train_{k}'] = -1.0
      writer.write_scalars(int(state.step), log_scalars)

      latest_checkpoint_manager.save(
          state.step,
          items={
              'train_state': jax.tree.map(np.array, state),
              'train_iter': train_iter,
          },
      )

      running_train_loss = 0.0
      running_train_loss_squared = 0.0
      running_train_step = 0
      log_dict = {
          'sum': collections.defaultdict(float),
          'count': collections.defaultdict(int),
      }
      train_start_time = time.time()

    if (
        state.step % config.eval_every_steps == 0
        or state.step == config.max_steps
        or state.step == 10
    ):

      val_start_time = time.time()
      running_val_loss = 0.0
      val_log_dict = {
          'sum': collections.defaultdict(float),
          'count': collections.defaultdict(int),
      }
      for i, batch in tqdm.tqdm(enumerate(val_dataloader), mininterval=10):
        # TODO(riegerfr): assert always same order/samples (no race conditions)
        batch = {
            'coord': batch['coord'],
            'feat': batch['feat'],
            '_dataset_index': batch.get('_dataset_index'),
        }
        assert (
            (-1 < batch['coord']) & (batch['coord'] < 1)
        ).mean() > 0.9, 'Coordinates should be in [-1, 1] after scaling.'

        val_loss, aux = p_val_step(state, batch, jax.random.fold_in(val_rng, i))
        for k, v in aux.items():
          if v is not None:
            val_log_dict['sum'][k] += v
            val_log_dict['count'][k] += 1

        running_val_loss += val_loss
        if i + 1 >= config.eval_steps:
          break
      val_loss = running_val_loss / (i + 1)  # pylint: disable=undefined-loop-variable
      logging.info(
          'Step: %d, Val Loss: %.5f, Time: %.2fs',
          state.step,
          val_loss,
          time.time() - val_start_time,
      )
      writer.write_scalars(
          int(state.step),
          {
              'val_loss': float(val_loss),
          },
      )
      log_scalars = {}
      for k, v in val_log_dict['sum'].items():
        if val_log_dict['count'][k] > 0:
          log_scalars[f'val_{k}'] = v / val_log_dict['count'][k]
        else:
          log_scalars[f'val_{k}'] = -1.0
      writer.write_scalars(int(state.step), log_scalars)
  variables = {'params': state.ema_params}
  if state.batch_stats:
    variables['batch_stats'] = state.batch_stats
  pred_init_feat = model.apply(
      variables,
      coord=init_coord,
      mutable=False,
      deterministic=True,
  )

  writer.close()
  latest_checkpoint_manager.wait_until_finished()
  best_checkpoint_manager.wait_until_finished()
  return pred_init_feat

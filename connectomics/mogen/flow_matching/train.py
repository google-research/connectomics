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
import time

from absl import logging
from clu import metric_writers
from connectomics.mogen.flow_matching import utils
from e3x.so3 import rotations
from etils import epath
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

from ffn.inference import storage

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

def train_flow_matching(
    config: ml_collections.ConfigDict, log_dir: str
) -> None | jax.Array:
  """Trains a flow matching model.

  Args:
    config: Configuration for training.
    log_dir: Directory to write logs.

  Returns:
    Generated samples after training.
  """
  workdir = epath.Path(config.workdir) / f'{config.name_str}/'
  log_dir = epath.Path(log_dir) / f'{config.name_str}/'
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
      dst_index_cond=config.train_set == 'mixed',
      add_dummy_cond=(len(config.initial_checkpoint_path) > 0),  # pylint: disable=g-explicit-length-test
      # TODO(riegerfr): make dst_index_cond cleaner
      # (i.e. no dependency on train set string here)
  )
  logging.debug('DEBUG: init_coord.shape = %s', init_coord.shape)
  logging.debug(
      'DEBUG: config.batch_size for assertion = %d', config.batch_size
  )
  assert init_coord.shape[1] == config.n_points
  assert init_coord.shape[2] == 3
  assert init_coord.shape[0] // config.num_devices == config.batch_size, (
      f'n_train_samples was {n_train_samples} but init_coord.shape[0] is'
      f' {init_coord.shape[0]} and config.batch_size is {config.batch_size} '
      f'and config.num_devices is {config.num_devices}'
  )

  if jax.process_index() == 0:
    with storage.atomic_file(
        str(workdir / 'init_data.npz'), 'wb'
    ) as f:
      save_dict = {'coord': np.asarray(init_coord)}
      if config.use_feat and init_feat is not None:
        save_dict['feat'] = np.asarray(init_feat)
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
      batch_stats=batch_stats,
      opt_state=optimizer.init(params),
      min_s_mmd_train=float('inf'),
  )

  if jax.process_index() == 0:
    with storage.atomic_file(
        str(workdir / 'init_coord.html'), 'w'
    ) as f:
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

  if latest_checkpoint_manager.latest_step():
    restored_data = latest_checkpoint_manager.restore(
        latest_checkpoint_manager.latest_step(),
        items={'train_state': state, 'train_iter': train_iter},
    )
    state = restored_data['train_state']
    train_iter = restored_data['train_iter']
    logging.info(
        'Restored checkpoint for step %d',
        latest_checkpoint_manager.latest_step(),
    )
  else:
    logging.info('Starting training from scratch.')
    if config.get('initial_checkpoint_path', ''):
      checkpoint_dir = epath.Path(config.initial_checkpoint_path)
      logging.info(
          'initial_checkpoint_path provided: %r', config.initial_checkpoint_path
      )
      initial_checkpoint_manager = ocp.CheckpointManager(
          directory=checkpoint_dir,
          checkpointers={
              'train_state': ocp.AsyncCheckpointer(
                  ocp.PyTreeCheckpointHandler()
              ),
          },
      )
      if initial_checkpoint_manager.latest_step() is not None:
        step_to_restore = initial_checkpoint_manager.latest_step()
        logging.info(
            'Loading initial weights from %r step %d',
            checkpoint_dir,
            step_to_restore,
        )
        restored_state = initial_checkpoint_manager.restore(
            step_to_restore,
            items={'train_state': state},
        )['train_state']
        if config.get('load_optimizer_state', True):
          state = state.replace(
              step=0,
              params=restored_state.params,
              ema_params=restored_state.ema_params,
              batch_stats=restored_state.batch_stats,
              opt_state=restored_state.opt_state,
              min_s_mmd_train=float('inf'),
          )
        else:
          state = state.replace(
              step=0,
              params=restored_state.params,
              ema_params=restored_state.ema_params,
              batch_stats=restored_state.batch_stats,
              min_s_mmd_train=float('inf'),
          )
        logging.info(
            'Restored initial weights from %r step %d',
            checkpoint_dir,
            step_to_restore,
        )
      else:
        logging.warning(
            'initial_checkpoint_path %r specified, but step %d not found in'
            ' %r. Checkpoint steps: %r',
            config.initial_checkpoint_path,
            initial_checkpoint_manager.latest_step(),
            checkpoint_dir,
            initial_checkpoint_manager.all_steps(),
        )
      train_iter = iter(train_dataloader)
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
  global_batch_size = config.batch_size * config.num_devices

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

    coord, feat, cond = utils.prep_data(
        batch,
        config.coord_scale,
        config.feat_scale,
        config.n_points // config.n_combine_samples,
        n_combine_samples=config.n_combine_samples,
        use_feat=config.use_feat,
        cond_mode=config.cond_mode,
        dst_index_cond=config.train_set == 'mixed',
        add_dummy_cond=len(config.initial_checkpoint_path) > 0,  # pylint: disable=g-explicit-length-test
    )

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
        reorder_noise_strength=config.reorder_noise_strength,
        point_cond_sample_threshold=config.point_cond_sample_threshold,
        feat_cond_dropout_threshold=config.feat_cond_dropout_threshold,
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

    coord, feat, cond = utils.prep_data(
        batch,
        config.coord_scale,
        config.feat_scale,
        config.n_points // config.n_combine_samples,
        n_combine_samples=config.n_combine_samples,
        use_feat=config.use_feat,
        cond_mode=config.cond_mode,
        dst_index_cond=config.train_set == 'mixed',
        add_dummy_cond=len(config.initial_checkpoint_path) > 0,  # pylint: disable=g-explicit-length-test
    )

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

  def generate(state, rng, cond, noise, point_cond_mask, guidance_scale, guide):
    """Generates samples from the model."""
    sample_shape = (init_coord.shape[1], init_coord.shape[-1] + feat_shape)
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
      ),
      out_shardings=batch_sharding,  # samples
      static_argnames=('guide',),
  )

  while jax.device_get(state.step) < config.max_steps:
    batch = next(train_iter)

    batch = {
        'coord': batch['coord'],
        'feat': batch['feat'],
        '_dataset_index': (
            batch['_dataset_index'] if '_dataset_index' in batch else None
        ),
    }
    assert ((-1 < batch['coord']) & (batch['coord'] < 1)).mean() > 0.9

    batch = jax.tree.map(jnp.array, batch)
    state, aux = p_train_step(state, batch)
    utils.log_bins(log_dict, aux)
    running_train_loss += aux['loss']
    running_train_loss_squared += aux['loss'] ** 2
    running_train_step += 1

    step = int(jax.device_get(state.step))

    if (
        step % config.log_train_every_steps == 0
        or step == config.max_steps
        or step == 10
        or step == 20
    ):
      train_loss = running_train_loss / running_train_step
      train_loss_variance = (
          running_train_loss_squared / running_train_step - train_loss**2
      )
      if jax.process_index() == 0:
        logging.info(
            'Step: %d, Train Loss: %.5f, Train Loss Variance: %.5f, Time:'
            ' %.2fs',
            step,
            train_loss,
            train_loss_variance,
            time.time() - train_start_time,
        )
      writer.write_scalars(step * global_batch_size, {'train_loss': train_loss})
      writer.write_scalars(
          step * global_batch_size,
          {'train_loss_variance': train_loss_variance},
      )
      writer.write_scalars(
          step * global_batch_size, utils.log_dict_to_scalars(log_dict)
      )

      latest_checkpoint_manager.save(
          step,
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
        step % config.eval_every_steps == 0
        or step == config.max_steps
        or step == 10
        or step == 20
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
      val_loss = running_val_loss / (i + 1)  # pylint: disable=undefined-loop-variable
      if jax.process_index() == 0:
        logging.info(
            'Step: %d, Val Loss: %.5f, Time: %.2fs',
            step,
            val_loss,
            time.time() - val_start_time,
        )
      writer.write_scalars(step * global_batch_size, {'val_loss': val_loss})

    if (
        step % config.generation_every_steps == 0
        or step == config.max_steps
        or step == 10
        or step == 20
    ):
      generation_start_time = time.time()
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
      x_gen_full = jnp.concatenate(
          [
              p_generate(
                  state,
                  jax.random.fold_in(generation_rng, i),
                  cond,
                  None,  # noise/x_0
                  point_cond_mask,
                  None,  # guidance_scale
                  False,  # guide
              )
              for i in range(max(1, config.n_samples // config.num_devices))
          ],
          axis=0,
      )[: config.n_samples]
      x_gen = x_gen_full[:, :, :3] / config.coord_scale
      if config.use_feat:
        x_feat_gen = x_gen_full[:, :, 3:] / config.feat_scale
      else:
        x_feat_gen = None

      assert x_gen.shape[0] == config.n_samples
      if not config.do_rotate:
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
        with storage.atomic_file(
            str(workdir / 'x_gen.html'), 'w'
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
      metrics_results = utils.compute_metrics(
          x_gen_gathered,
          config,
      )

      (
          s_mmd_train,
          s_mmd_val,
          s_fid_train,
          s_fid_val,
          s_mmd_train_sub,
          s_mmd_val_sub,
      ) = metrics_results

      logging.info(
          'computed metrics, s_mmd_train: %r, s_mmd_val: %r',
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
          ) as f:
            save_dict = {'coord': np.asarray(x_gen_gathered)}
            if config.use_feat and x_feat_gen_gathered is not None:
              save_dict['feat'] = np.asarray(x_feat_gen_gathered)
            np.savez_compressed(f, **save_dict)

          with storage.atomic_file(
              str(workdir / 'best_s_mmd_train.html'), 'w'
          ) as f:
            utils.plot_point_clouds(
                x_gen_gathered[: utils.N_PLOT],
                n_combine_samples=config.n_combine_samples,
            ).write_html(f)

        if config.point_cond > 0:
          utils.log_point_cond(
              state,
              cond,
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
                'train_state': jax.tree.map(np.array, state),
            },
        )

      writer.write_scalars(
          step * global_batch_size,
          {
              'mmd_train': mmd_train,
              'mmd_val': mmd_val,
              'fid_train': fid_train,
              'fid_val': fid_val,
              's_mmd_train': s_mmd_train_sub,
              's_mmd_val': s_mmd_val_sub,
              's_fid_train': s_fid_train,
              's_fid_val': s_fid_val,
              's_mmd_train_mst': s_mmd_train,
              's_mmd_val_mst': s_mmd_val,
              'generation_time': metrics_start_time - generation_start_time,
              'metrics_time': time.time() - metrics_start_time,
          },
      )
      logging.info(
          'Step: %d, MMD Train: %.5f, MMD Val: %.5f, Min s_MMD Train: %.5f',
          step,
          mmd_train,
          mmd_val,
          state.min_s_mmd_train,
      )
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
        logging.info('Stopping training at because of high MMD')
        break

  writer.close()
  latest_checkpoint_manager.wait_until_finished()
  best_checkpoint_manager.wait_until_finished()
  return best_x_gen

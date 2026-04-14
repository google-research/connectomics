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
"""Generates samples from a trained Flow Matching model."""

from collections.abc import Sequence
import json
import time

from absl import app
from absl import flags
from absl import logging
from connectomics.jax import training
from etils import epath
import jax
from jax import sharding
import jax.numpy as jnp
import ml_collections
import ml_collections.config_flags
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm

from connectomics.mogen.flow_matching import utils
from ffn.inference import storage

FLAGS = flags.FLAGS

training.define_training_flags()


def _get_num_samples_in_file(output_filepath: epath.Path) -> int:
  """Returns the number of samples in an existing file, or -1 if not found/invalid."""
  try:
    with gfile.Open(output_filepath, 'rb') as f:
      with np.load(f) as data:
        return data['arr_0'].shape[0]
  except Exception:  # pylint: disable=broad-except
    logging.warning(
        'File %s exists but cannot be loaded, regenerating.',
        output_filepath,
        exc_info=True,
    )
    return -1


def main(_: Sequence[str]):
  training.prep_training()

  config = FLAGS.config

  # Sharding for parallel inference
  work_unit_id = config.get('work_unit_id', 0)

  num_work_units = config.num_work_units

  original_seed = config.seed

  if num_work_units > 1:
    logging.info(
        'Running in sharded mode. Work unit %d of %d.',
        work_unit_id,
        num_work_units,
    )
    config.seed += work_unit_id
    config.n_samples_total = config.n_samples_total // num_work_units

  train_workdir = epath.Path(config.train_workdir)

  found_subdir = next(
      (
          subdir
          for subdir in train_workdir.iterdir()
          if subdir.is_dir() and (subdir / 'config.json').exists()
      ),
      None,
  )

  train_workdir = found_subdir if found_subdir is not None else train_workdir
  config_path = train_workdir / 'config.json'

  output_dir = (
      epath.Path(config.output_dir)
      if config.output_dir
      else train_workdir / f'inference_original_seed{original_seed}'
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  logging.info('Loading train config from %s', config_path)
  with config_path.open('r') as f:
    train_config_dict = json.load(f)
    train_config = ml_collections.config_dict.ConfigDict(train_config_dict)

  n_combine_samples = train_config.get('n_combine_samples', 1)

  model_rng, gen_rng = jax.random.split(jax.random.key(config.seed))

  assert (
      not train_config.use_feat
  ), 'Inference script currently does not support features.'

  fake_batch = {
      'coord': jnp.zeros((
          train_config.batch_size * n_combine_samples,
          train_config.n_points,
          3,
      )),
      'feat': None,
  }
  if train_config.train_set == 'mixed':
    # the mixed dataset has classes
    fake_batch['_dataset_index'] = jnp.zeros((
        train_config.batch_size * n_combine_samples,
    ))

  init_coord, init_feat, init_cond = utils.prep_data(
      fake_batch,
      train_config.coord_scale,
      train_config.feat_scale,
      train_config.n_points // n_combine_samples,
      n_combine_samples=n_combine_samples,
      use_feat=train_config.use_feat,
      cond_mode=train_config.cond_mode,
      dst_index_cond=train_config.train_set == 'mixed',
      add_dummy_cond=(config.get('extend_point_cond', False)),
  )

  model_cond = None
  if init_cond is not None:
    model_cond = jnp.concatenate(
        (init_cond[:1], jnp.zeros_like(init_cond[:1])), axis=1
    )

  point_cond_mask_for_init = None
  if train_config.get('extend_point_cond', False):
    # for old models (without point_cond support)
    point_cond_mask_for_init = jnp.zeros((1, init_coord.shape[1]), dtype=bool)
  m_model, params, batch_stats = utils.get_model(
      model_rng,
      init_coord=init_coord[:1],
      init_feat=init_feat[:1] if init_feat is not None else None,
      config=train_config,
      cond=model_cond,
      point_cond_mask=point_cond_mask_for_init,
      n_combine_samples=n_combine_samples,
  )

  optimizer = utils.get_optimizer(train_config)(
      learning_rate=train_config.lr, weight_decay=train_config.wd
  )
  if train_config.clip_grad > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config.clip_grad), optimizer
    )
  opt_state = optimizer.init(params)  # for the dummy state
  dummy_state = utils.TrainState(
      step=0,
      params=params,
      ema_params=params,
      batch_stats=batch_stats,
      opt_state=opt_state,
      min_s_mmd_train=float('inf'),
  )

  checkpoint_dir = train_workdir / 'best_checkpoints'
  checkpoint_manager = ocp.CheckpointManager(
      directory=checkpoint_dir,
      checkpointers={
          'train_state': ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
      },
  )

  latest_step = checkpoint_manager.latest_step()
  # TODO(riegerfr): add option to load from specific step

  logging.info('Restoring checkpoint from step %d', latest_step)
  state = checkpoint_manager.restore(
      latest_step, items={'train_state': dummy_state}
  )['train_state']

  if train_config.get('extend_point_cond', False):
    init_cond = jnp.concatenate(
        (
            init_cond,
            jnp.zeros((init_cond.shape[0], 1), dtype=init_cond.dtype),
        ),
        axis=1,
    )

  mesh = sharding.Mesh(np.array(jax.devices()), ('batch',))
  replicate_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec())
  batch_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec('batch'))
  state = jax.device_put(state, replicate_sharding)

  feat_shape = init_feat.shape[-1] if init_feat is not None else 0

  def generate(state, rng, cond, noise, point_cond_mask):
    """Generates samples."""
    sample_shape = (init_coord.shape[1], init_coord.shape[-1] + feat_shape)
    batch_size = train_config.batch_size * jax.device_count()
    return (
        utils.generate_samples(
            m_model,
            state,
            batch_size,
            sample_shape,
            rng,
            train_config.sample_steps,
            train_config.sample_schedule,
            cond=cond,
            noise=noise,
            point_cond_mask=point_cond_mask,
            guidance_scale=None,
            guide=False,
        )[:, :, :3]
        / train_config.coord_scale
    )

  p_generate = jax.jit(
      generate,
      in_shardings=(
          replicate_sharding,
          replicate_sharding,
          batch_sharding,
          batch_sharding,
          batch_sharding,
      ),
      out_shardings=batch_sharding,
  )

  cond_for_gen = None
  if init_cond is not None:
    cond_for_gen = jnp.concatenate(
        (jnp.zeros_like(init_cond), jnp.zeros_like(init_cond)), axis=-1
    )
    cond_for_gen = jnp.tile(cond_for_gen, (jax.device_count(), 1))
    cond_for_gen = jax.device_put(cond_for_gen, batch_sharding)

  point_cond_mask_for_gen = None
  if train_config.get('extend_point_cond', False):
    # for old models (without point_cond support)
    batch_size_gen = train_config.batch_size * jax.device_count()
    point_cond_mask_for_gen = jnp.zeros(
        (batch_size_gen, init_coord.shape[1]), dtype=bool
    )
    point_cond_mask_for_gen = jax.device_put(
        point_cond_mask_for_gen, batch_sharding
    )

  logging.info('Starting generation.')
  start_time = time.time()
  n_generated = 0
  n_files = 0
  samples_to_save = []

  with tqdm.tqdm(total=config.n_samples_total) as pbar:
    while n_generated < config.n_samples_total:
      output_filepath = output_dir / f'samples_{work_unit_id}_{n_files:05d}.npz'

      num_samples_in_file = _get_num_samples_in_file(output_filepath)
      if num_samples_in_file == config.n_samples_per_file:
        logging.info(
            'File %s already exists, skipping generation.', output_filepath
        )

        current_samples_count = 0
        while current_samples_count < config.n_samples_per_file:
          gen_rng, step_rng = jax.random.split(gen_rng)
          current_samples_count += train_config.batch_size * jax.device_count()

        n_generated += current_samples_count
        n_files += 1
        pbar.update(current_samples_count)
        samples_to_save = []
        continue

      gen_rng, step_rng = jax.random.split(gen_rng)

      x_gen = p_generate(
          state,
          step_rng,
          cond_for_gen,
          None,
          point_cond_mask_for_gen,
      )

      samples_to_save.append(np.array(x_gen))
      current_samples_count = sum(s.shape[0] for s in samples_to_save)

      if current_samples_count >= config.n_samples_per_file:
        all_samples = np.concatenate(samples_to_save, axis=0)
        with storage.atomic_file(str(output_filepath), 'wb') as f:
          np.savez_compressed(f, all_samples)
        n_generated += all_samples.shape[0]
        n_files += 1
        pbar.update(all_samples.shape[0])
        logging.info(
            'Saved %d samples to %s', all_samples.shape[0], output_filepath
        )
        samples_to_save = []

  total_time = time.time() - start_time
  logging.info(
      'Finished generation of %d samples in %.2f seconds.',
      n_generated,
      total_time,
  )


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)

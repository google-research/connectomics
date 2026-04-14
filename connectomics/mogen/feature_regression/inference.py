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
"""Generates samples from a trained feature regression model."""

from collections.abc import Sequence
import json
import time

from absl import app
from absl import flags
from absl import logging
from connectomics.jax import training
from connectomics.mogen.flow_matching import utils
from etils import epath
import jax
from jax import sharding
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm

from ffn.inference import storage

FLAGS = flags.FLAGS


training.define_training_flags()


def main(_: Sequence[str]):
  training.prep_training()

  config = FLAGS.config

  work_unit_id = config.get('work_unit_id', 0)

  num_work_units = config.num_work_units

  if num_work_units > 1:
    logging.info(
        'Running in sharded mode. Work unit %d of %d.',
        work_unit_id,
        num_work_units,
    )

  train_workdir = epath.Path(config.train_workdir)

  found_subdir = next(
      (
          subdir
          for subdir in train_workdir.iterdir()
          if subdir.is_dir() and (subdir / 'config.json').exists()
      ),
      None,
  )

  train_workdir = found_subdir
  config_path = train_workdir / 'config.json'

  logging.info('Loading train config from %s', config_path)
  with config_path.open('r') as f:
    train_config_dict = json.load(f)
    train_config = ml_collections.config_dict.ConfigDict(train_config_dict)

  output_dir = epath.Path(config.inoutput_dir + '_feats/')
  input_dir = epath.Path(config.inoutput_dir)
  logging.info('Loading samples from %s', input_dir)
  logging.info('Saving samples to %s', output_dir)

  model_rng = jax.random.key(0)

  batch_size = train_config.batch_size * jax.device_count()

  n_combine_samples = train_config.get('n_combine_samples', 1)

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

  init_coord, _, _ = utils.prep_data(
      fake_batch,
      train_config.coord_scale,
      train_config.feat_scale,
      train_config.n_points // n_combine_samples,
      n_combine_samples=n_combine_samples,
      use_feat=False,
      dst_index_cond=train_config.train_set == 'mixed',
  )

  m_model, params, batch_stats = utils.get_model(
      model_rng,
      init_coord=init_coord[:1],
      init_feat=None,
      config=train_config,
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
  )

  checkpoint_dir = train_workdir / 'checkpoints'
  checkpoint_manager = ocp.CheckpointManager(
      directory=checkpoint_dir,
      checkpointers={
          'train_state': ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
      },
  )

  latest_step = checkpoint_manager.latest_step()

  logging.info('Restoring checkpoint from step %d', latest_step)
  state = checkpoint_manager.restore(
      latest_step, items={'train_state': dummy_state}
  )['train_state']

  mesh = sharding.Mesh(np.array(jax.devices()), ('batch',))
  replicate_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec())
  batch_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec('batch'))
  state = jax.device_put(state, replicate_sharding)

  variables = {'params': state.ema_params}
  if state.batch_stats:
    variables['batch_stats'] = state.batch_stats

  def generate(coord: jax.Array) -> jax.Array:
    """Generates samples."""
    pred_feat = m_model.apply(
        variables,
        coord=coord,
        mutable=False,
        deterministic=True,
    )
    return pred_feat

  p_generate = jax.jit(
      generate,
      in_shardings=(batch_sharding,),
      out_shardings=batch_sharding,
  )
  logging.info('Starting generation.')
  start_time = time.time()

  all_files_to_read = list(input_dir.iterdir())

  worker_files = [
      f
      for f in all_files_to_read
      if f.name.startswith(f'samples_{work_unit_id}_')
  ]

  for file in tqdm.tqdm(
      worker_files if num_work_units > 1 else all_files_to_read,
      desc=f'Work unit {work_unit_id} files',
  ):
    logging.info('Processing file %s', file)

    output_filepath = output_dir / f'feat_{file.name}'

    if output_filepath.exists():

      logging.info(
          'File %s already exists, skipping generation.', output_filepath
      )

      continue
    with gfile.Open(file, 'rb') as f:
      input_points = np.load(f)['arr_0']

    feat_pred_list = []
    for i in tqdm.tqdm(
        range(input_points.shape[0] // batch_size),
        desc=f'Work unit {work_unit_id} batches',
    ):
      feat_pred_list.append(
          p_generate(input_points[i * batch_size : (i + 1) * batch_size])
      )

    all_samples = np.concatenate(feat_pred_list, axis=0)
    with storage.atomic_file(str(output_filepath), 'wb') as f:
      np.savez_compressed(f, all_samples)
    logging.info(
        'Saved %d samples to %s', all_samples.shape[0], output_filepath
    )

  total_time = time.time() - start_time
  logging.info(
      'Finished generation in %.2f seconds.',
      total_time,
  )


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)

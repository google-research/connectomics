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
"""Main file for morphology generation with flow matching."""

from collections.abc import Sequence
import os

os.environ['XLA_FLAGS'] = (
    '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    ' inter_op_parallelism_threads=1 NPROC=1'
)
# Limiting the number of threads to avoid xmanager failures.
# https://github.com/jax-ml/jax/discussions/22739#discussioncomment-10204864

import logging  # pylint: disable=g-import-not-at-top, g-bad-import-order

from absl import app
from absl import flags
from connectomics.jax import training
import jax

from connectomics.mogen.flow_matching import train

FLAGS = flags.FLAGS

training.define_training_flags()


def main(argv: Sequence[str] = ('',)) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  training.prep_training()
  logging.info('config %s, workdir %s', FLAGS.config, FLAGS.workdir)
  train.train_flow_matching(config=FLAGS.config, log_dir=FLAGS.workdir)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)

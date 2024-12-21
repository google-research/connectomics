# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Utilities for model checkpointing."""

import re
from typing import Any, Optional, Sequence, TypeVar

from clu import checkpoint as checkpoint_lib
from etils import epath
import flax
import grain.python as grain
from orbax import checkpoint as ocp
import tensorflow as tf


T = TypeVar('T')


class MixedMultihostCheckpoint(checkpoint_lib.MultihostCheckpoint):
  """Like MultihostCheckpoint, but with a single source of FLAX weights.

  TF settings are restored per-host as in the base class.

  This prevents the model from loading potentially inconsistent weights
  saved by other hosts. Weights might be inconsistent when they are saved
  based on wall-clock time instead of step count.
  """

  def load_state(
      self, state: Optional[T], checkpoint: Optional[str] = None
  ) -> T:
    flax_path = self._flax_path(self._checkpoint_or_latest(checkpoint))
    flax_path = re.sub('checkpoints-[0-9]*', 'checkpoints-0', flax_path)
    if not tf.io.gfile.exists(flax_path):
      raise FileNotFoundError(f'Checkpoint {checkpoint} does not exist')
    with tf.io.gfile.GFile(flax_path, 'rb') as f:
      return flax.serialization.from_bytes(state, f.read())


def get_checkpoint_manager(
    workdir: epath.PathLike,
    item_names: Sequence[str],
) -> ocp.CheckpointManager:
  """Returns a checkpoint manager."""
  checkpoint_dir = epath.Path(workdir) / 'checkpoints'
  return ocp.CheckpointManager(
      checkpoint_dir,
      item_names=item_names,
      options=ocp.CheckpointManagerOptions(
          create=True, cleanup_tmp_directories=True),
  )


def save_checkpoint(
    manager: ocp.CheckpointManager,
    state: Any,
    step: int,
    pygrain_checkpointers: Sequence[str] = ('train_iter',),
    wait_until_finished: bool = True,
):
  """Saves a checkpoint.

  Args:
    manager: Checkpoint manager to use.
    state: Data to be saved.
    step: Step at which to save the data.
    pygrain_checkpointers: Names of items for which to use pygrain checkpointer.
    wait_until_finished: If True, blocks until checkpoint is written.
  """
  save_args_dict = {}
  for k, v in state.items():
    if k in pygrain_checkpointers:
      save_args_dict[k] = grain.PyGrainCheckpointSave(v)
    else:
      save_args_dict[k] = ocp.args.StandardSave(v)
  manager.save(step, args=ocp.args.Composite(**save_args_dict))
  if wait_until_finished:
    manager.wait_until_finished()


def restore_checkpoint(
    manager: ocp.CheckpointManager,
    state: Any,
    step: int | None = None,
    pygrain_checkpointers: Sequence[str] = ('train_iter',),
) -> Any:
  """Restores a checkpoint.

  Args:
    manager: Checkpoint manager to use.
    state: Data to be restored.
    step: Step at which to save the data. If None, uses latest step.
    pygrain_checkpointers: Names of items for which to use pygrain checkpointer.

  Returns:
    Restored data.
  """
  restore_args_dict = {}
  for k, v in state.items():
    if k in pygrain_checkpointers:
      restore_args_dict[k] = grain.PyGrainCheckpointRestore(v)
    else:
      restore_args_dict[k] = ocp.args.StandardRestore(v)
  return manager.restore(
      manager.latest_step() if step is None else step,
      args=ocp.args.Composite(**restore_args_dict))

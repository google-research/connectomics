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
"""Utilities for training JAX / FLAX models."""

from collections.abc import Mapping
import time
from typing import Any, Optional

from absl import flags
from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

FLAGS = flags.FLAGS

PyTree = Any


def get_optimizer(
    config: ml_collections.ConfigDict,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
  """Returns an optax optimizer based on standard config settings."""
  if config.scale_learning_rate_by_global_batch_size:
    try:
      global_batch_size = config.global_batch_size
    except AttributeError:
      global_batch_size = config.per_device_batch_size * jax.device_count()
    init_lr = global_batch_size * config.learning_rate
    min_lr = global_batch_size * config.min_learning_rate
  else:
    init_lr = config.learning_rate
    min_lr = config.min_learning_rate

  if config.scheduler == "linear":
    lr = optax.linear_schedule(init_lr, min_lr, config.num_train_steps)
  elif config.scheduler == "cosine":
    alpha_min = min_lr / init_lr
    lr = optax.cosine_decay_schedule(init_lr, config.num_train_steps, alpha_min)
  elif config.scheduler == "cosine_warmup":
    lr = optax.warmup_cosine_decay_schedule(
        init_value=min_lr,
        peak_value=init_lr,
        warmup_steps=config.num_warmup_steps,
        decay_steps=config.num_train_steps - config.num_warmup_steps,
        end_value=min_lr
    )
  else:
    lr = optax.constant_schedule(init_lr)

  txs = []
  if config.weight_decay and config.optimizer != "adamw":
    txs.append(optax.add_decayed_weights(config.weight_decay))

  grad_clip_method = config.get("grad_clip_method", "")
  grad_clip_value = config.get("grad_clip_value", None)
  if grad_clip_method and grad_clip_value is not None:
    if grad_clip_method == "clip":
      txs.append(optax.clip(grad_clip_value))
    elif grad_clip_method == "clip_by_block_rms":
      txs.append(optax.clip_by_block_rms(grad_clip_value))
    elif grad_clip_method == "clip_by_global_norm":
      txs.append(optax.clip_by_global_norm(grad_clip_value))
    elif grad_clip_method == "adaptive_grad_clip":
      txs.append(optax.adaptive_grad_clip(grad_clip_value))
    else:
      raise ValueError(f"{grad_clip_method} is not supported.")

  if config.optimizer == "sgd":
    txs.append(optax.sgd(lr))
  elif config.optimizer == "momentum":
    txs.append(optax.sgd(lr, nesterov=True, momentum=config.sgd_momentum))
  elif config.optimizer == "adam":
    txs.append(optax.adam(lr))
  elif config.optimizer == "lamb":
    txs.append(optax.lamb(lr, weight_decay=config.weight_decay))
  elif config.optimizer == "adamw":
    txs.append(
        optax.adamw(
            lr,
            weight_decay=config.weight_decay,
            eps=config.get("adam_eps", default=1e-8),
        )
    )
  else:
    raise ValueError(f"Unknown optimizer type: {config.optimizer}.")

  return optax.chain(*txs), lr


class ReportProgress(periodic_actions.ReportProgress):
  """Helper to report training progress."""

  def __init__(
      self,
      batch_size: int,
      *,
      num_train_steps: Optional[int] = None,
      writer: Optional[metric_writers.MetricWriter] = None,
      every_steps: Optional[int] = None,
      every_secs: Optional[float] = 60.0,
  ):
    """Constructor.

    Args:
      batch_size: number of training examples in the batch
      num_train_steps: max. number of training steps the model will train for
      writer: MetricWriter object
      every_steps: how often to report the progress in number of training steps
      every_secs: how often to report progress as time interval
    """
    super().__init__(
        num_train_steps=num_train_steps,
        writer=writer,
        every_steps=every_steps,
        every_secs=every_secs,
    )
    self.batch_size = batch_size

  def _apply(self, step: int, t: float):
    steps_per_sec = (step - self._previous_step) / (t - self._previous_time)
    message = f"{steps_per_sec:.1f} steps/s"
    if self._num_train_steps:
      eta_seconds = (self._num_train_steps - step) / steps_per_sec
      message += (
          f", {100 * step / self._num_train_steps:.1f}% @{step}, "
          f"ETA: {eta_seconds / 60:.0f} min"
      )
    if self._time_per_part:
      total = time.time() - self._t0
      message += " ({:.0f} min : {})".format(
          total / 60,
          ", ".join(
              f"{100 * dt / total:.1f}% {name}"
              for name, dt in sorted(self._time_per_part.items())
          ),
      )
    # This should be relatively cheap so we can do it in the main thread.
    # However, this RPC sometimes fails and can crash the training loop,
    # so we ignore any exceptions.
    try:
      platform.work_unit().set_notes(message)
    except Exception:  # pylint:disable=broad-exception-caught
      logging.exception("Failed to set XM notes.")
      pass

    if self._writer is not None:
      data = {
          "steps_per_sec": steps_per_sec,
          "examples_per_sec": steps_per_sec * self.batch_size,
      }
      self._writer.write_scalars(step, data)


class StepTraceContextHelper:
  """Helper class to use jax.profiler.StepTraceContext.

  Allows direct iteration over a tf.data dataset within the context.
  """

  def __init__(self, name: str, init_step_num: int):
    self.name = name
    self.step_num = init_step_num
    self.context = None

  def __enter__(self):
    self.context = jax.profiler.StepTraceAnnotation(
        self.name, step_num=self.step_num
    )
    self.step_num += 1
    self.context.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    assert self.context is not None, "Exited context without entering."
    self.context.__exit__(exc_type, exc_value, tb)
    self.context = None

  def next_step(self):
    if self.context is None:
      raise ValueError("Must call next_step() within a context.")
    self.__exit__(None, None, None)
    self.__enter__()


def define_training_flags():
  """Defines standard flags used for model training."""
  ml_collections.config_flags.DEFINE_config_file(
      "config", None, "Training configuration.", lock_config=True
  )
  flags.DEFINE_string("workdir", None, "Work unit directory.")
  flags.mark_flags_as_required(["config", "workdir"])
  flags.DEFINE_string(
      "service_address", None, "The address of the tf.data service."
  )


def prep_training():
  """Logs system info, and hides accelerators from TF."""
  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.set_visible_devices([], "GPU")

  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax_xla_backend = (
        "None" if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
    )
    logging.info("Using JAX XLA backend %s", jax_xla_backend)

  logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())
  logging.info(
      "JAX num devices: local: %d  global: %d",
      jax.local_device_count(),
      jax.device_count(),
  )

  # Add a note so that we can tell which Borg task is which JAX host.
  # (Borg task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f"host: {jax.process_index()}/{jax.process_count()}"
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
  )


def minmax_params(params: PyTree) -> tuple[PyTree, PyTree]:
  """Returns the min and max (across workers) of every parameter."""
  min_pars = jax.lax.pmin(params, axis_name="batch")
  max_pars = jax.lax.pmax(params, axis_name="batch")
  return min_pars, max_pars


def check_state(p_minmax: ..., params: PyTree, step: int):
  """Verifies that parameters are consistent between all workers."""
  min_state, max_state = p_minmax(params=params)
  if not jax.tree_util.tree_all(
      jax.tree_util.tree_map(lambda a, b: jnp.all(a == b), min_state, max_state)
  ):
    raise ValueError(
        f"Found inconsistency between state on different workers at step {step}"
    )


def get_rng(seed: None | int | tuple[int, int]) -> jax.Array:
  """Returns a JAX RNGKey."""
  if seed is None:
    # Case 1: No random seed given, use XManager ID.
    # All processes (and restarts) get exactly the same seed but every work unit
    # and experiment is different.
    work_unit = platform.work_unit()
    rng = (work_unit.experiment_id, work_unit.id)
  elif isinstance(seed, int):
    # Case 2: Single integer given.
    rng = (0, seed)
  else:
    # Case 3: tuple[int, int] given.
    if not isinstance(seed, (tuple, list)) or len(seed) != 2:
      raise ValueError(
          "Random seed must be an integer or tuple of 2 integers "
          f"but got {seed!r}"
      )
    rng = seed
  return jnp.asarray(rng).astype(jnp.uint32)  # shape: [2]


def reshape_batch_local_devices(
    batch: Mapping[str, Any]
) -> Mapping[str, np.ndarray]:
  """Reshapes a batch to have the leading dimension for the local devices."""
  leading_dims = [jax.local_device_count(), -1]
  return jax.tree.map(
      lambda x: np.reshape(x, leading_dims + list(x.shape[1:])), batch
  )


class MeasureTime:
  """Measures execution times."""

  def __init__(
      self, store: Mapping[str, list[float]], name: str, log: bool = True
  ):
    """Constructor.

    Args:
      store: execution time will be appended to store[name]
      name: a string ID for this context
      log: whether to log the duration of every execution
    """
    self.name = name
    self.store = store
    self.log = log

  def __enter__(self):
    self.start = time.perf_counter()
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.time = time.perf_counter() - self.start
    self.store[self.name].append(self.time)
    if self.log:
      logging.info("time[%s] = %.4f", self.name, self.time)

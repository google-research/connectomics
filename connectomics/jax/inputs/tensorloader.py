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
"""DataLoader that relies mainly on a tensorstore to load batches of data.

The code is adapted from pygrain.DataLoader and is significantly faster when
loading entire batches from tensorstore instead of the per-item convention with
subsequent stacking using numpy. Parallelization of loading chunks is handled
by tensorstore and the loader only requests batches using threading, which
allows for tensorstore caching and has shown to be faster and more memory-
efficient than using shared memory and multiprocessing.

Sharding for distributed loading as well as sampling use pygrain.
"""

import collections
from collections.abc import Iterator
import dataclasses
from typing import Any, Optional, Sequence, Type, TypeVar

import grain.python as pygrain
import numpy as np
import tensorstore as ts

from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T', bound='BatchMetadata')
Batch = dict[str, Any] | Any

# Import required dataclasses to make tensorloader a central import
NoSharding = pygrain.NoSharding
ShardOptions = pygrain.ShardOptions
ShardByJaxProcess = pygrain.ShardByJaxProcess


@dataclasses.dataclass(slots=True)
class BatchMetadata:
  """BatchMetadata contains metadata about a batch of records.

  BatchMetadata are usually created from a sequence of Metadata emitted by
  a pygrain Sample and contain read indices, which indicate steps, record_keys
  to read from the TensorSource and rng keys for eventual randomness required.
  """

  indices: Sequence[int]
  rngs: Optional[Sequence[np.random.Generator]] = None

  @classmethod
  def from_entries(
      cls: Type[T], records: Sequence[pygrain.RecordMetadata]
  ) -> T:
    indices = [record.record_key for record in records]
    rngs = [record.rng for record in records]
    return cls(indices, rngs)


class TensorSource:
  """TensorSource protocol that loads batches of data from a TensorStore.

  TODO(aleximmer): consider adding transforms or index transforms.
  """

  def __len__(self) -> int:
    raise NotImplementedError()

  def __getitem__(self, metadata: BatchMetadata) -> Batch:
    raise NotImplementedError()

  @property
  def item_shape(self) -> dict[str, tuple[int, ...]]:
    """Return shape of individual items."""
    raise NotImplementedError()


class BasicTensorSource(TensorSource):
  """Tensor source where the leading dimension corresponds to data points."""

  def __init__(self, ts_spec: dict[str, Any]):
    self._data = ts.open(ts_spec).result()

  def __len__(self) -> int:
    return self._data.shape[0]

  def __getitem__(self, metadata: BatchMetadata) -> Batch:
    return self._data[metadata.indices]


class TensorLoader:
  """TensorLoader loads batches from tensorstore data source.

  In comparison to grain, does not support operations but batches automatically.
  Tensorstore is significantly faster than numpy stacking for large arrays and
  can be optimized using custom chunk sizes that align with batches.
  """

  def __init__(
      self,
      *,
      tensor_source: TensorSource,
      batch_size: int = 1,
      sampler: pygrain.Sampler | None = None,
      num_epochs: int | None = None,
      shuffle: bool | None = None,
      seed: int | None = None,
      shard_options: ShardOptions | None = None,
      num_threads: int = 8,
      initial_batch: int = 0,
  ):
    """Loads and transforms input data.

    Args:
      tensor_source: Responsible for retrieving batches of records based on
        their indices.
      batch_size: Number of examples to jointly query from the data source.
      sampler: Custom sampler, defaults to pygrain.IndexSampler.
      num_epochs: Number of epochs to yield data for. Passed to sampler and
        defaults to 1.
      shuffle: Whether to randomly or sequentially index the tensor_source.
        Passed to sampler and defaults to False.
      seed: Random seed for sampler.
      shard_options: Options for how data should be sharded when using multiple
        machines (~ JAX processes) and data parallelism.
      num_threads: Number of threads for parallel prefetching of batches.
      initial_batch: Batch number to start from (use for checkpointing).
    """
    super().__init__()
    self._tensor_source = tensor_source
    self._batch_size = batch_size
    if shard_options is None:
      shard_options = NoSharding()
    if sampler is not None:
      if any([num_epochs, shuffle, seed]):
        raise ValueError(
            'Cannot specify sampler and one of num_epochs, shuffle, and seed.'
        )
      assert not any([num_epochs, shuffle, seed])
      self._sampler = sampler
    else:
      shuffle = False if shuffle is None else shuffle
      num_epochs = 1 if num_epochs is None else num_epochs
      self._sampler = pygrain.IndexSampler(
          num_records=len(tensor_source),
          shard_options=shard_options,
          shuffle=shuffle,
          num_epochs=num_epochs,
          seed=seed,
      )
    if num_threads <= 0:
      raise ValueError(f'num_threads must be positive: {num_threads}')
    self._num_threads = num_threads
    self._shard_options = shard_options
    if initial_batch < 0:
      raise ValueError(f'initial_batch must be positive: {initial_batch}')
    self.set_initial_batch(initial_batch)

  def set_initial_batch(self, initial_batch: int):
    # start negative and advance within __iter__
    self._initial_step = (
        initial_batch * self._batch_size
        - self._shard_options.shard_count
        + self._shard_options.shard_index
    )

  def __iter__(self) -> Iterator[Batch]:
    """Read sampled record indices to load and yield batches."""
    next_index = self._initial_step + self._shard_options.shard_count
    buffer = collections.deque()

    def make_index_batch(next_index_: int) -> tuple[int, list[int]]:
      next_indices = []
      for _ in range(self._batch_size):
        next_indices.append(next_index_)
        next_index_ += self._shard_options.shard_count
      return next_index_, next_indices

    def prefetch_elements(indices: Sequence[int]) -> Any:
      metadata = []
      for i, index in enumerate(indices):
        try:
          metadata.append(self._sampler[index])
        except IndexError as e:
          if i == 0 or self._shard_options.drop_remainder:
            raise e
          else:
            break
      batch_metadata = BatchMetadata.from_entries(metadata)
      data = self._tensor_source[batch_metadata]
      return data

    with ThreadPoolExecutor(self._num_threads) as executor:
      # Fill the buffer initially.
      while len(buffer) < self._num_threads:
        next_index, batch_indices = make_index_batch(next_index)
        buffer.append(executor.submit(prefetch_elements, batch_indices))

      # Iterate until IndexError from the Sampler.
      while True:
        try:
          batch = buffer.popleft().result()
        except IndexError:
          return
        yield batch
        next_index, batch_indices = make_index_batch(next_index)
        buffer.append(executor.submit(prefetch_elements, batch_indices))

  @property
  def tensor_source(self) -> TensorSource:
    return self._tensor_source

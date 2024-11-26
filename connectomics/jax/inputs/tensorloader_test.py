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
"""Tests for tensorloader."""

from absl.testing import absltest  # pylint: disable=g-multiple-import
from absl.testing import parameterized  # pylint: disable=g-multiple-import
from connectomics.jax.inputs import tensorloader as tl
import grain.python as pygrain
import numpy as np
import tensorstore as ts


class TestTensorSource(tl.TensorSource):
  """Testing source that works for in-memory non-persistent tensorstore."""

  def __init__(self, data: ts.TensorStore):
    self._data = data

  def __len__(self) -> int:
    return self._data.shape[0]

  def __getitem__(self, metadata: tl.BatchMetadata) -> tl.Batch:
    return {'data': self._data[metadata.indices].read().result()}


class TensorloaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    ts_test_config = {
        'create': True,
        'driver': 'zarr',
        'dtype': 'float64',
        'metadata': {'shape': [16, 8, 2]},
        'kvstore': {'driver': 'memory'},
    }
    self.data = ts.open(ts_test_config).result()
    generator = np.random.RandomState(seed=4321)
    self.example_data = generator.randn(*self.data.shape).astype(np.float64)
    self.data[...] = self.example_data

  def test_completeness(self):
    loader = tl.TensorLoader(
        tensor_source=TestTensorSource(self.data),
        batch_size=4,
        num_epochs=1,
        shuffle=False,
    )
    data = np.vstack([batch['data'] for batch in loader])
    np.testing.assert_array_equal(data, self.example_data)

  def test_incomplete_last_batch(self):
    loader = tl.TensorLoader(
        tensor_source=TestTensorSource(self.data),
        shard_options=tl.ShardOptions(0, 1, drop_remainder=False),
        batch_size=3,
        num_epochs=1,
        shuffle=False,
    )
    data = np.vstack([batch['data'] for batch in loader])
    np.testing.assert_array_equal(data, self.example_data)

  def test_skip_last_batch(self):
    loader = tl.TensorLoader(
        tensor_source=TestTensorSource(self.data),
        shard_options=tl.ShardOptions(0, 1, drop_remainder=True),
        batch_size=3,
        num_epochs=1,
        shuffle=False,
    )
    data = np.vstack([batch['data'] for batch in loader])
    np.testing.assert_array_equal(data, self.example_data[:-1])

  def test_checkpointing(self):
    loader = tl.TensorLoader(
        tensor_source=TestTensorSource(self.data),
        batch_size=4,
        num_epochs=1,
        shuffle=False,
        initial_batch=2,
    )
    data = np.vstack([batch['data'] for batch in loader])
    np.testing.assert_array_equal(data, self.example_data[8:])

  def test_sharding_complement(self):
    base_config = dict(
        tensor_source=TestTensorSource(self.data),
        batch_size=4,
        num_epochs=1,
    )
    shard_options_a = tl.ShardOptions(shard_index=0, shard_count=2)
    shard_options_b = tl.ShardOptions(shard_index=1, shard_count=2)
    loader_a = tl.TensorLoader(**base_config, shard_options=shard_options_a)
    loader_b = tl.TensorLoader(**base_config, shard_options=shard_options_b)
    batches = []
    for batch_a, batch_b in zip(loader_a, loader_b):
      batches.append(np.vstack([batch_a['data'], batch_b['data']]))
    data = np.vstack(batches)
    self.assertEqual(data.shape, self.example_data.shape)
    # test statistic. sampler separates regions and is not globally sequential.
    self.assertEqual(np.mean(data), np.mean(self.example_data))

  def test_basic_source(self):
    ts_test_config = {
        'driver': 'array',
        'dtype': 'int32',
        'array': np.arange(16).reshape((4, 4)),
    }
    tsource = tl.BasicTensorSource(ts_test_config)
    metadata = tl.BatchMetadata(indices=[0, 1])
    batch = tsource[metadata]
    np.testing.assert_array_equal(batch, np.arange(8).reshape((2, 4)))

  def test_external_sampler(self):
    shard_options = tl.ShardOptions(shard_index=0, shard_count=1)
    source = TestTensorSource(self.data)
    sampler = pygrain.SequentialSampler(len(source), shard_options)
    loader = tl.TensorLoader(
        tensor_source=source, shard_options=shard_options, sampler=sampler
    )
    data = np.vstack([batch['data'] for batch in loader])
    np.testing.assert_array_equal(data, self.example_data)


if __name__ == '__main__':
  absltest.main()

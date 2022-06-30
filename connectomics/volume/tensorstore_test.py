# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Tests for tensorstore volumes."""

from absl import flags
from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.volume import tensorstore as tsv
import numpy as np
import numpy.testing as npt

FLAGS = flags.FLAGS
BBox = bounding_box.BoundingBox


def default_data(size):
  data = np.ones(shape=size, dtype='uint32')
  rank = len(data.shape)
  data[(0,) * rank] = 0
  data = np.cumsum(data.ravel()).reshape(data.shape)
  return data


class TensorstoreTest(absltest.TestCase):

  def test_volume_metadata(self):
    metadata = tsv.TensorstoreMetadata(
        bounding_boxes=[BBox([0, 0, 0], [10, 11, 12])],
        voxel_size=(8, 8, 33),
    )
    as_json = """
    {
      "voxel_size": [8,8,33],
      "bounding_boxes": [{
        "start": [0,0,0],
        "size": [10,11,12]
      }]
    }
    """
    self.assertEqual(metadata, tsv.TensorstoreMetadata.from_json(as_json))

  def test_creation(self):
    bbox = BBox([0, 0, 0], [10, 11, 12])
    metadata = tsv.TensorstoreMetadata(
        bounding_boxes=[bbox],
        voxel_size=(8, 8, 33),
    )
    data = default_data([1, 12, 11, 10])
    spec = {
        'driver': 'array',
        'dtype': str(data.dtype),
        'array': data,
    }

    vol = tsv.TensorstoreVolume(tsv.TensorstoreConfig(spec, metadata))
    self.assertEqual((8, 8, 33), vol.voxel_size)
    self.assertEqual(4, vol.ndim)
    self.assertEqual(np.uint64, vol.dtype)
    self.assertEqual((10, 11, 12), vol.volume_size)
    self.assertSequenceEqual([bbox], vol.bounding_boxes)

  def test_metadata_mismatch(self):
    # Bad TS shape
    data = np.random.uniform(size=[50, 60, 70])
    metadata = tsv.TensorstoreMetadata(
        bounding_boxes=[BBox([0, 0, 0], [10, 11, 12])],
        voxel_size=(8, 8, 33),
    )
    spec = {
        'driver': 'array',
        'dtype': str(data.dtype),
        'array': data,
    }
    with self.assertRaises(ValueError):
      tsv.TensorstoreVolume(tsv.TensorstoreConfig(spec, metadata))

    # Bad voxel_size
    data = np.random.uniform(size=[1, 50, 60, 70])
    metadata = tsv.TensorstoreMetadata(
        bounding_boxes=[BBox([0, 0, 0], [10, 11, 12])],
        voxel_size=(8, 8, 0),
    )
    spec = {
        'driver': 'array',
        'dtype': str(data.dtype),
        'array': data,
    }
    with self.assertRaises(ValueError):
      tsv.TensorstoreVolume(tsv.TensorstoreConfig(spec, metadata))

    # Bad bounding boxes
    data = np.random.uniform(size=[1, 50, 60, 70])
    metadata = tsv.TensorstoreMetadata(
        bounding_boxes=[],
        voxel_size=(8, 8, 0),
    )
    spec = {
        'driver': 'array',
        'dtype': str(data.dtype),
        'array': data,
    }
    with self.assertRaises(ValueError):
      tsv.TensorstoreVolume(tsv.TensorstoreConfig(spec, metadata))

  def test_tensorstore_array_creation(self):
    metadata = tsv.TensorstoreMetadata(
        bounding_boxes=[BBox([0, 0, 0], [10, 11, 12])],
        voxel_size=(8, 8, 33),
    )
    vol = tsv.TensorstoreArrayVolume(default_data([1, 12, 11, 10]), metadata)
    self.assertEqual((8, 8, 33), vol.voxel_size)
    self.assertEqual((10, 11, 12), vol.volume_size)

  def test_tensorstore_access(self):
    metadata = tsv.TensorstoreMetadata(
        bounding_boxes=[BBox([0, 0, 0], [10, 11, 12])],
        voxel_size=(8, 8, 33),
    )
    data = default_data([1, 12, 11, 10])
    vol = tsv.TensorstoreArrayVolume(data, metadata)

    index_exp = np.s_[:, 2:7, :, 8:]
    npt.assert_array_equal(data[index_exp], vol[index_exp].data)


if __name__ == '__main__':
  absltest.main()

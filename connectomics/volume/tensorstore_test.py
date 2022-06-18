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

import os

from absl import flags
from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.common import file
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
    metadata = tsv.TensorstoreVolumeMetadata(
        bounding_boxes=[BBox([0, 0, 0], [10, 11, 12])],
        voxel_size=(8, 8, 33),
    )
    self.assertEqual(
        metadata, tsv.TensorstoreVolumeMetadata.from_json(metadata.to_json()))

  def test_creation(self):
    bbox = BBox([0, 0, 0], [10, 11, 12])
    metadata = tsv.TensorstoreVolumeMetadata(
        bounding_boxes=[bbox],
        voxel_size=(8, 8, 33),
    )
    data = default_data([1, 12, 11, 10])
    spec = {
        'driver': 'array',
        'dtype': str(data.dtype),
        'array': data,
    }

    # Test with metadata object
    vol = tsv.TensorstoreVolume(spec, metadata)
    self.assertEqual((8, 8, 33), vol.voxel_size)
    self.assertEqual(4, vol.ndim)
    self.assertEqual(np.uint64, vol.dtype)
    self.assertEqual((10, 11, 12), vol.volume_size)
    self.assertSequenceEqual([bbox], vol.bounding_boxes)

    # Test with serialized metadata object
    vol = tsv.TensorstoreVolume(spec, metadata.to_json())
    self.assertEqual((8, 8, 33), vol.voxel_size)
    self.assertSequenceEqual([bbox], vol.bounding_boxes)

    # Test with malformed serialized metadata object
    with self.assertRaises(ValueError):
      tsv.TensorstoreVolume(spec, 'totally not json')

    # Test with metadata from file
    tmp_dir = FLAGS.test_tmpdir
    metadata_path = os.path.join(tmp_dir, 'metadata.json')
    with file.Open(metadata_path, 'w') as f:
      f.write(metadata.to_json())
    vol = tsv.TensorstoreVolume(spec, metadata_path)
    self.assertEqual((8, 8, 33), vol.voxel_size)
    self.assertSequenceEqual([bbox], vol.bounding_boxes)

    # Test with malformed metadata file
    metadata_path = os.path.join(tmp_dir, 'metadata_malformed.json')
    with open(metadata_path, 'w') as f:
      f.write('no json here')
    with self.assertRaises(ValueError):
      tsv.TensorstoreVolume(spec, metadata_path)

    # Test with nonexistant metadata file
    with self.assertRaises(ValueError):
      tsv.TensorstoreVolume(
          spec, '/if/this/file/exists/and/is/valid/Ill/eat/my/hat.json')

  def test_tensorstore_array_creation(self):
    metadata = tsv.TensorstoreVolumeMetadata(
        bounding_boxes=[BBox([0, 0, 0], [10, 11, 12])],
        voxel_size=(8, 8, 33),
    )
    vol = tsv.TensorstoreArrayVolume(default_data([1, 12, 11, 10]), metadata)
    self.assertEqual((8, 8, 33), vol.voxel_size)
    self.assertEqual((10, 11, 12), vol.volume_size)

  def test_tensorstore_access(self):
    metadata = tsv.TensorstoreVolumeMetadata(
        bounding_boxes=[BBox([0, 0, 0], [10, 11, 12])],
        voxel_size=(8, 8, 33),
    )
    data = default_data([1, 12, 11, 10])
    vol = tsv.TensorstoreArrayVolume(data, metadata)

    index_exp = np.s_[:, 2:7, :, 8:]
    npt.assert_array_equal(data[index_exp], vol[index_exp])


if __name__ == '__main__':
  absltest.main()

"""Tests for base."""

from absl.testing import absltest
from connectomics.volume import base


class BaseVolumeTest(absltest.TestCase):

  def test_not_implemented(self):
    v = base.BaseVolume()
    with self.assertRaises(NotImplementedError):
      _ = v[1]

    for field in [
        'volume_size', 'voxel_size', 'shape', 'ndim', 'dtype', 'bounding_boxes'
    ]:
      with self.assertRaises(NotImplementedError):
        _ = getattr(v, field)

if __name__ == '__main__':
  absltest.main()

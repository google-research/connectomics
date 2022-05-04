"""Tests for base."""

from absl.testing import absltest
from connectomics.common import array
from connectomics.volume import base


class BaseVolumeTest(absltest.TestCase):

  def test_not_implemented(self):
    v = base.BaseVolume()

    for field in [
        'volume_size', 'voxel_size', 'shape', 'ndim', 'dtype', 'bounding_boxes'
    ]:
      with self.assertRaises(NotImplementedError):
        _ = getattr(v, field)

  def test_get_points(self):
    tself = self

    class ShimVolume(base.BaseVolume):

      def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        self.called = False

      @property
      def shape(self) -> array.Tuple4i:
        return (1, 12, 11, 10)

      def get_points(self, points):
        self.called = True
        tself.assertLen(points, 4)
        for i in range(1, 4):
          tself.assertLen(points[i], 3)

    v = ShimVolume()
    _ = v[0, (1, 2, 3), (4, 5, 6), (7, 8, 9)]
    self.assertTrue(v.called)

    v = ShimVolume()
    _ = v[0, [1, 2, 3], [4, 5, 6], [7, 8, 9]]
    self.assertTrue(v.called)

  def test_get_slices(self):
    tself = self

    expected = (
        slice(0, 1, None),
        slice(1, 3, 1),
        slice(5, 11, 1),
        slice(0, 10, 1),
    )

    class ShimVolume(base.BaseVolume):

      def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        self.called = False

      @property
      def shape(self) -> array.Tuple4i:
        return (1, 12, 11, 10)

      def get_slices(self, slices):
        self.called = True
        tself.assertLen(slices, 4)
        tself.assertEqual(expected, slices)

    v = ShimVolume()
    _ = v[0, 1:3, 5:, :]
    self.assertTrue(v.called)


if __name__ == '__main__':
  absltest.main()

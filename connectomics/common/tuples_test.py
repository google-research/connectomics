# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tuples."""

from absl.testing import absltest
from connectomics.common import tuples


class NamedTupleTest(absltest.TestCase):

  def test_xyz_zyx(self):
    x, y, z = [1, 2, 3]
    xyz = tuples.XYZ(x, y, z)
    zyx = tuples.ZYX(z, y, x)

    for tup in [xyz, zyx]:
      self.assertEqual(tup.x, x)
      self.assertEqual(tup.y, y)
      self.assertEqual(tup.z, z)

    self.assertEqual(xyz, zyx)

    self.assertEqual(xyz.xyz, xyz)
    self.assertEqual(xyz.zyx, zyx)
    self.assertEqual(xyz.zyx.xyz, xyz)

    self.assertEqual(zyx.zyx, zyx)
    self.assertEqual(zyx.xyz, xyz)
    self.assertEqual(zyx.xyz.zyx, zyx)

    self.assertEqual(xyz[0], x)
    self.assertEqual(xyz[1], y)
    self.assertEqual(xyz[2], z)

    self.assertEqual(zyx[0], z)
    self.assertEqual(zyx[1], y)
    self.assertEqual(zyx[2], x)

    self.assertEqual(xyz, (x, y, z))
    self.assertEqual(zyx, (z, y, x))

  def test_xyzc_czyx(self):
    x, y, z, c = [1, 2, 3, 4]
    xyz = tuples.XYZ(x, y, z)
    zyx = tuples.ZYX(z, y, x)
    xyzc = tuples.XYZC(x, y, z, c)
    czyx = tuples.CZYX(c, z, y, x)

    for tup in [xyzc, czyx]:
      self.assertEqual(tup.x, x)
      self.assertEqual(tup.y, y)
      self.assertEqual(tup.z, z)
      self.assertEqual(tup.c, c)

    self.assertEqual(xyzc, czyx)

    self.assertEqual(xyzc.xyz, xyz)
    self.assertEqual(xyzc.zyx, zyx)
    self.assertEqual(xyzc.xyzc, xyzc)
    self.assertEqual(xyzc.czyx, czyx)

    self.assertEqual(czyx, xyzc)

    self.assertEqual(czyx.xyz, xyz)
    self.assertEqual(czyx.zyx, zyx)
    self.assertEqual(czyx.czyx, czyx)
    self.assertEqual(czyx.xyzc, xyzc)

    self.assertEqual(xyzc[0], x)
    self.assertEqual(xyzc[1], y)
    self.assertEqual(xyzc[2], z)
    self.assertEqual(xyzc[3], c)

    self.assertEqual(czyx[0], c)
    self.assertEqual(czyx[1], z)
    self.assertEqual(czyx[2], y)
    self.assertEqual(czyx[3], x)

    self.assertEqual(xyzc, (x, y, z, c))
    self.assertEqual(czyx, (c, z, y, x))


if __name__ == '__main__':
  absltest.main()

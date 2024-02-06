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

"""Tests for io_utils."""

import tempfile

from absl.testing import absltest
from connectomics.common import io_utils


class ObjectUtilsTest(absltest.TestCase):

  def test_load_equivalences(self):
    equivs = ((1, 2), (3, 4), (2, 5))

    with tempfile.NamedTemporaryFile('wt') as tmp:
      for a, b in equivs:
        tmp.write('%d,%d\n' % (a, b))

      tmp.flush()
      uf, graph = io_utils.load_equivalences([tmp.name])

      self.assertEqual(uf.Find(1), uf.Find(5))
      self.assertEqual(uf.Find(3), uf.Find(4))
      self.assertEqual(graph.number_of_nodes(), 5)
      self.assertEqual(graph.number_of_edges(), 3)


if __name__ == '__main__':
  absltest.main()

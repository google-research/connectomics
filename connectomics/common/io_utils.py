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

"""Utilities for object info."""

from typing import Sequence

from absl import logging
import networkx as nx
import numpy as np
import pandas as pd

from . import file
from . import union_find


def load_equivalences(
    paths: Sequence[str],
) -> tuple[union_find.UnionFind, nx.Graph]:
  """Loads equivalences from a text file.

  Args:
    paths: sequence of paths to the text files of equivalences; id0,id1 per
      line, or id0,id1,x,y,z.

  Returns:
    tuple of:
      UnionFind object representing the equivalences
      NX graph object representing the equivalences
  """
  uf = union_find.UnionFind()
  equiv_graph = nx.Graph()

  for path in paths:
    with file.Path(path).open("rt") as f:
      reader = pd.read_csv(
          f, sep=",", engine="c", comment="#", chunksize=4096, header=None
      )
      for chunk in reader:
        if len(chunk.columns) not in (2, 5):
          logging.fatal(
              "Unexpected # of columns (%d), want 2 or 5", len(chunk.columns)
          )

        edges = chunk.values[:, :2]
        equiv_graph.add_edges_from(edges)
        for id_a, id_b in edges:
          uf.Union(id_a, id_b)

  return uf, equiv_graph


def load_relabel_map(path: str) -> dict[int, int]:
  """Loads a label map from a text file."""
  with file.Path(path).open("r") as f:
    df = pd.read_csv(f, engine="c", comment="#", header=None, dtype=np.uint64)
    return {int(a): int(b) for a, b in df.to_numpy(dtype=np.uint64)}


def load_object_list(path: str) -> set[int]:
  """Loads an object list from a text file."""
  with file.Path(path).open("r") as f:
    df = pd.read_csv(f, engine="c", comment="#", header=None, dtype=np.uint64)
    return set(int(x) for x in df.to_numpy(dtype=np.uint64).ravel())

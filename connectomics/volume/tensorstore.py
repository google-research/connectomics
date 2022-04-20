"""A Tensorstore-backed Volume."""

from typing import List

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.volume import base
import numpy as np


class TensorstoreVolume(base.BaseVolume):
  """Tensorstore-backed Volume."""
  # TODO(timblakely): Implement.

  def __getattr__(self, attr):
    raise NotImplementedError

  def __getitem__(self, ind):
    raise NotImplementedError

  @property
  def volume_size(self) -> array.Tuple3i:
    raise NotImplementedError

  @property
  def shape(self) -> array.Tuple4i:
    raise NotImplementedError

  @property
  def ndim(self) -> int:
    raise NotImplementedError

  @property
  def dtype(self) -> np.dtype:
    raise NotImplementedError

  @property
  def bounding_boxes(self) -> List[bounding_box.BoundingBox]:
    return []


"""Volume decorators for modifying volumes on-the-fly."""

from typing import Sequence, Union

from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.volume import base
import numpy as np


class VolumeDecorator(base.BaseVolume):
  """Delegates to wrapped volumes, optionally applying transforms."""

  wrapped: base.BaseVolume

  def __init__(self, wrapped: base.BaseVolume):
    self._wrapped = wrapped

  def __getitem__(self, ind):
    """Delegate to wrapped BaseVolume by default."""
    return self._wrapped[ind]

  @property
  def volume_size(self) -> array.Tuple3i:
    return self._wrapped.volume_size

  @property
  def voxel_size(self) -> array.Tuple3i:
    return self._wrapped.voxel_size

  @property
  def shape(self) -> array.Tuple4i:
    return self._wrapped.shape

  @property
  def ndim(self) -> int:
    return self._wrapped.ndim

  @property
  def dtype(self) -> np.dtype:
    return self._wrapped.dtype

  @property
  def bounding_boxes(self) -> list[bounding_box.BoundingBox]:
    return self._wrapped.bounding_boxes


class ScaleChannels(VolumeDecorator):
  """Scales channel values by given factors."""

  def __init__(self, factors: Sequence[Union[int, float]],
               wrapped: base.BaseVolume):
    """Initialize the wrapper with per-channel scale factors.

    Args:
      factors: Sequence with len equal to num_channels giving the scale factors
        by which to multiply each data channel.
      wrapped: Wrapped volume to apply scaling to.

    Raises:
      ValueError: If len(factors) != num_channels.
    """
    super(ScaleChannels, self).__init__(wrapped)
    if len(factors) != self._wrapped.shape[0]:
      raise ValueError('len(factors) must equal number of channels (shape[0]): '
                       f'{factors} vs. {self._wrapped.shape[0]}')

    # Make 4d so easy to broadcast.
    self.factors = np.reshape(factors, (self._wrapped.shape[0], 1, 1, 1))

  def __getitem__(self, slc):
    if len(slc) == 3:
      # Special case for VolumeStore meaning all channels.  This is different
      # from Numpy behavior.
      factors = self.factors
    else:
      factors = self.factors[slc[0], ...]
      if factors.ndim == 3:
        # If slc[0] was a single channel index, and got auto-squeezed, put it
        # back.
        factors.shape = 1, 1, 1, 1
    data = self._wrapped[slc]
    return (data * factors).astype(data.dtype)

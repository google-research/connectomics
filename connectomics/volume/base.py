"""4-D volume abstraction."""

import dataclasses
import functools
from typing import List, Optional, Dict, Any

from connectomics.common import array
from connectomics.common import bounding_box
import dataclasses_json
import numpy as np


@dataclasses_json.dataclass_json
@functools.partial(dataclasses.dataclass, frozen=True)
class VolumeDescriptor:
  """De/Serializable description of a volume."""
  # List of python dicts of specs to decorate the volume via
  # volume_decorator.from_specs.  E.g.:
  #   '[{"decorator": "ZSub", "args": [{"2198": 2197}]}]'
  # If left unspecified, the undecorated volume is used.
  decorator_specs: List[Dict[str,
                             Any]] = dataclasses.field(default_factory=list)

  # Exactly one of `volumeinfo` or `tensorstore_config` must be specified
  # JSON tensorstore configuration.
  tensorstore_config: Optional[Dict[str, Any]] = None

  # Internal use only. Path to the VolumeInfo file..
  volinfo: Optional[str] = None


# TODO(timblakely): Make generic-typed so it exposes both VolumeInfo and
# Tensorstore via .descriptor.
class BaseVolume:
  """Common interface to multiple volume backends for Decorators."""

  def __getitem__(self, ind):
    raise NotImplementedError

  @property
  def volume_size(self) -> array.Tuple3i:
    raise NotImplementedError

  @property
  def voxel_size(self) -> array.Tuple3i:
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

  # TODO(timblakely): determine what other attributes we want to make mandatory for
  # all implementations and add them here.

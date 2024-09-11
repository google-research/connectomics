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
"""Type-safe tuple and NamedTuple utilities."""

import dataclasses
from typing import Any, Callable, Generic, NamedTuple, Type, TypeVar
import dataclasses_json

T = TypeVar('T', int, float)
C = TypeVar('C')


class XYZ(Generic[T], NamedTuple):
  """XYZ is a named tuple for a 3-dimensional vector.

  Allows static type checker to differentiate between XYZ and ZYX, and allows
  switching between the two via named properties.
  """

  x: T
  y: T
  z: T

  def __eq__(self, other):
    if isinstance(other, XYZ) or isinstance(other, ZYX):
      return self.x == other.x and self.y == other.y and self.z == other.z
    # Defer to tuple equality.
    return self[:] == other

  @property
  def xyz(self) -> 'XYZ[T]':
    return self

  # Allow swizzling into ZYX format.
  @property
  def zyx(self) -> 'ZYX[T]':
    return ZYX(*self[::-1])


class ZYX(Generic[T], NamedTuple):
  """ZYX is a named tuple for a 3-dimensional vector.

  Allows static type checker to differentiate between XYZ and ZYX, and allows
  switching between the two via named properties.
  """

  z: T
  y: T
  x: T

  # Allow swizzling into XYZ format.
  @property
  def xyz(self) -> 'XYZ[T]':
    return XYZ(*self[::-1])

  @property
  def zyx(self) -> 'ZYX[T]':
    return self

  def __eq__(self, other):
    if isinstance(other, XYZ) or isinstance(other, ZYX):
      return self.x == other.x and self.y == other.y and self.z == other.z
    # Defer to tuple equality.
    return self[:] == other


class XYZC(Generic[T], NamedTuple):
  """XYZC is a named tuple for a 4-dimensional vector."""

  x: T
  y: T
  z: T
  c: T

  def __eq__(self, other):
    if isinstance(other, XYZC) or isinstance(other, CZYX):
      return (
          self.x == other.x
          and self.y == other.y
          and self.z == other.z
          and self.c == other.c
      )
    # Defer to tuple equality.
    return self[:] == other

  @property
  def xyz(self) -> 'XYZ[T]':
    return XYZ(self.x, self.y, self.z)

  @property
  def zyx(self) -> 'ZYX[T]':
    return ZYX(self.z, self.y, self.x)

  @property
  def xyzc(self) -> 'XYZC[T]':
    return self

  @property
  def czyx(self) -> 'CZYX[T]':
    return CZYX(*self[::-1])


class CZYX(Generic[T], NamedTuple):
  """CZYX is a named tuple for a 4-dimensional vector."""

  c: T
  z: T
  y: T
  x: T

  # Allow swizzling into XYZ format.
  @property
  def xyz(self) -> 'XYZ[T]':
    return XYZ(self.x, self.y, self.z)

  @property
  def zyx(self) -> 'ZYX[T]':
    return ZYX(self.z, self.y, self.x)

  @property
  def xyzc(self) -> 'XYZC[T]':
    return XYZC(*self[::-1])

  @property
  def czyx(self) -> 'CZYX[T]':
    return self

  def __eq__(self, other):
    if isinstance(other, XYZC) or isinstance(other, CZYX):
      return (
          self.x == other.x
          and self.y == other.y
          and self.z == other.z
          and self.c == other.c
      )
    # Defer to tuple equality.
    return self[:] == other


def named_tuple_field(
    cls: C,
    encoder: Callable[..., Any] | None = None,
    decoder: Callable[..., C] | Type[C] | None = tuple,
):
  """Add metadata to allow NamedTuple decoding in dataclasses.

  Example usage:
    @dataclass
    class Foo:
      location: XYZ[float] = named_tuple_field(XYZ)
      dest_voxel: CZYX[float] = named_tuple_field(CZYX)

  Args:
    cls: The NamedTuple class to use.
    encoder: The encoder to use for the NamedTuple.
    decoder: The decoder to use for the NamedTuple.

  Returns:
    A dataclass field that will decode to the given NamedTuple.
  """
  return dataclasses.field(
      metadata={
          'named_tuple_type': cls,
          **dataclasses_json.config(encoder=encoder, decoder=decoder),
      }
  )


@dataclasses.dataclass(frozen=True)
class DataclassWithNamedTuples:
  """Parent class that allows dataclasses to have NamedTuple members.

  Subclass to allow dataclasses to accept generic constructor arguments,
  ensuring runtime NamedTuples.
  """

  def __post_init__(self):
    for field in dataclasses.fields(self):
      named_tuple_type = field.metadata.get('named_tuple_type', None)
      if not named_tuple_type:
        continue
      # Use object.__setattr__, since setattr won't work on frozen dataclasses.
      object.__setattr__(
          self, field.name, named_tuple_type(*getattr(self, field.name))
      )

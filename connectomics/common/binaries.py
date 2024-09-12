# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Binary executable encapsulation for connectomics pipelines."""

import dataclasses
import pathlib
from typing import Any, Sequence

from connectomics.common import file


FlagDict = dict[str, Any]
FlagTypes = dict[str, bool | str | int | float | pathlib.Path]


def flags_str(flags: Sequence[str] | None = None) -> str:
  if flags is None:
    return ''
  return ' '.join(flags_array(flags))


def flags_array(flags: FlagTypes | None = None) -> list[str]:
  flag_pairs = []
  if flags is None:
    return []
  for k, v in flags.items():
    if isinstance(v, bool):
      flag_pairs.append(k)
    elif isinstance(v, pathlib.Path):
      flag_pairs.append(f'{k}={v.as_posix()}')
    else:
      flag_pairs.append(f'{k}={v}')
  return [f'--{v}' for v in flag_pairs]


@dataclasses.dataclass(frozen=True)
class ExternalBinary:
  """A binary that can be run on the local machine or executed remotely."""

  path: file.PathLike

  local: bool = False
  flags: dict[str, Any] | None = None
  flag_overrides: dict[str, Any] | None = None

  def make_flags(self, default_flags: FlagDict | None = None) -> FlagDict:
    flags = self.flags if self.flags is not None else {}
    if default_flags is not None:
      flags.update(default_flags)
    return flags

  @property
  def full_path(self) -> pathlib.Path:
    return pathlib.Path(self.path)

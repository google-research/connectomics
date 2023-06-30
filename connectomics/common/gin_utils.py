# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Utilities for gin configurations."""

from typing import Sequence, Union

import gin


@gin.configurable
def list_to_str(
    strings: Sequence[Union[float, int, str]], separator: str = '_') -> str:
  """Join strings.

  Usage example:
    filename = @list_to_str()
    list_to_str.strings = ["example", "seed", %seed]
    list_to_str.separator = "_"

  Args:
    strings: List of strings.
    separator: Separator used between strings.

  Returns:
    Joined string.
  """
  return separator.join([str(i) for i in strings])


@gin.configurable
def integer_division(dividend: int, divisor: int) -> int:
  """Integer division.

  Usage example:
    size = @integer_division()
    integer_division.dividend = 128
    integer_division.divisor = %divisor

  Args:
    dividend: Dividend.
    divisor: Divisor.

  Returns:
    Result of integer division.
  """
  return dividend // divisor

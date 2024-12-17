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
"""Utilties for grain.

Code for `all_ops` and `parse` are forked from scenic --
this implementation uses `grain.python` rather than `grain.tensorflow`.
"""

import ast
import dataclasses
import inspect
import re
import sys
from typing import Any, Optional, Sequence, Type

import grain.python as grain
import numpy as np

FlatFeatures = dict[str, Any]

# Regex that finds upper case characters.
_CAMEL_CASE_RGX = re.compile(r'(?<!^)(?=[A-Z])')


@dataclasses.dataclass(frozen=False)
class ClipValues(grain.MapTransform):
  """Clips values between `min` an `max`.

  Attr:
    keys: Keys to apply the transformation to.
    min_value: Minimum value.
    max_value: Maximum value.
  """

  keys: str | Sequence[str] = ('x',)
  min_value: float = -1.0
  max_value: float = 1.0

  def __post_init__(self):
    assert self.min_value < self.max_value
    self.keys = (self.keys,) if isinstance(self.keys, str) else self.keys

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for k in self.keys:
      if k not in features: continue
      arr = features[k].astype(np.float32)
      features[k] = np.clip(arr, self.min_value, self.max_value)
    return features


@dataclasses.dataclass(frozen=False)
class ExpandDims(grain.MapTransform):
  """Expands the shape of an array.

  Attr:
    keys: Keys to apply the transformation to.
    axis: Position for placement.
  """

  keys: str | Sequence[str] = ('x',)
  axis: int | Sequence[int] = 0

  def __post_init__(self):
    self.keys = (self.keys,) if isinstance(self.keys, str) else self.keys

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for k in self.keys:
      if k not in features: continue
      features[k] = np.expand_dims(features[k], axis=self.axis)
    return features


@dataclasses.dataclass(frozen=False)
class PadValues(grain.MapTransform):
  """Pads values.

  Attr:
    keys: Keys to apply the transformation to.
    pad_width: Padding width.
    mode: Padding mode.
  """

  keys: str | Sequence[str] = ('x',)
  pad_width: int | Sequence[int] = 0
  mode: str = 'constant'

  def __post_init__(self):
    self.keys = (self.keys,) if isinstance(self.keys, str) else self.keys

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for k in self.keys:
      if k not in features: continue
      features[k] = np.pad(
          features[k], pad_width=self.pad_width, mode=self.mode)
    return features


@dataclasses.dataclass(frozen=False)
class RescaleValues(grain.MapTransform):
  """Rescales values from `min/max_input` to `min/max_output`.

  Attr:
    keys: Keys to apply the transformation to.
    min_input: The minimum value of the input.
    max_input: The maximum value of the input.
    min_output: The minimum value of the output.
    max_output: The maximum value of the output.
  """

  keys: str | Sequence[str] = ('x',)
  min_output: float = 0.0
  max_output: float = 1.0
  min_input: float = 0.0
  max_input: float = 255.0

  def __post_init__(self):
    assert self.min_output < self.max_output
    assert self.min_input < self.max_input
    self.keys = (self.keys,) if isinstance(self.keys, str) else self.keys

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for k in self.keys:
      if k not in features: continue
      arr = features[k].astype(np.float32)
      arr = (arr - self.min_input) / (self.max_input - self.min_input)
      arr = self.min_output + arr * (self.max_output - self.min_output)
      features[k] = arr
    return features


@dataclasses.dataclass(frozen=False)
class ReshapeValues(grain.MapTransform):
  """Reshapes values.

  Attr:
    keys: Keys to apply the transformation to.
    newshape: New shape.
  """

  keys: str | Sequence[str] = ('x',)
  newshape: int | Sequence[int] = -1

  def __post_init__(self):
    self.keys = (self.keys,) if isinstance(self.keys, str) else self.keys

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for k in self.keys:
      if k not in features: continue
      features[k] = features[k].reshape(self.newshape)
    return features


@dataclasses.dataclass(frozen=False)
class ShiftAndDivideValues(grain.MapTransform):
  """Subtracts shift from values and divides by scaling factor.

  Attr:
    keys: Keys to apply the transformation to.
    shift: Shift to subtract.
    divisor: Scale factor to divide by.
  """

  keys: str | Sequence[str] = ('x',)
  shift: float = 0.0
  divisor: float = 1.0

  def __post_init__(self):
    assert self.divisor != 0.0
    self.keys = (self.keys,) if isinstance(self.keys, str) else self.keys

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for k in self.keys:
      if k not in features: continue
      arr = features[k].astype(np.float32)
      arr = (arr - self.shift) / self.divisor
      features[k] = arr
    return features


@dataclasses.dataclass(frozen=False)
class TransposeValues(grain.MapTransform):
  """Transposes values.

  Attr:
    keys: Keys to apply the transformation to.
    axis: If specified, it must be a tuple or list which contains a permutation.
      If not specified, which reverses order of axes.
  """

  keys: str | Sequence[str] = ('x',)
  axis: Optional[Sequence[int]] = None

  def __post_init__(self):
    self.keys = (self.keys,) if isinstance(self.keys, str) else self.keys

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for k in self.keys:
      if k not in features: continue
      features[k] = features[k].transpose(self.axis)
    return features


def get_all_ops(module_name: str) -> list[
    tuple[str, Type[grain.Transformation]]]:
  """Helper to return all preprocess ops in a module.

  Modules that define processing ops can simply define:
  all_ops = lambda: process_spec.get_all_ops(__name__)
  all_ops() will then return a list with all dataclasses being
  grain.Transformation.

  Args:
    module_name: Name of the module. The module must already be imported.

  Returns:
    List of tuples of process ops. The first tuple element is the class name
    converted to snake case (MyAwesomeTransform => my_awesome_transform) and
    the second element is the class.
  """
  transforms = [grain.MapTransform, grain.RandomMapTransform,
                grain.FilterTransform]

  def is_op(x) -> bool:
    return (inspect.isclass(x)
            and dataclasses.is_dataclass(x)
            and any(issubclass(x, t) for t in transforms))

  op_name = lambda n: _CAMEL_CASE_RGX.sub('_', n).lower()
  members = inspect.getmembers(sys.modules[module_name])
  return [(op_name(name), op) for name, op in members if is_op(op)]


def _get_op_class(
    expr: list[ast.stmt],
    available_ops: dict[str, type[grain.Transformation]]
    ) -> Type[grain.Transformation]:
  """Gets the process op fn from the given expression."""
  if isinstance(expr, ast.Call):
    fn_name = expr.func.id
  elif isinstance(expr, ast.Name):
    fn_name = expr.id
  else:
    raise ValueError(
        f'Could not parse function name from expression: {expr!r}.')
  if fn_name in available_ops:
    return available_ops[fn_name]
  raise ValueError(
      f'"{fn_name}" is not available (available ops: {list(available_ops)}).')


def _parse_single_preprocess_op(
    spec: str,
    available_ops: dict[str, Type[grain.Transformation]]
    ) -> grain.Transformation:
  """Parsing the spec for a single preprocess op.

  The op can just be the method name or the method name followed by any
  arguments (both positional and keyword) to the method.
  See the test cases for some valid examples.

  Args:
    spec: String specifying a single processing operations.
    available_ops: Available preprocessing ops.

  Returns:
    The Transformation corresponding to the spec.
  """
  try:
    expr = ast.parse(spec, mode='eval').body  # pytype: disable=attribute-error
  except SyntaxError as e:
    raise ValueError(f'{spec!r} is not a valid preprocess op spec.') from e
  op_class = _get_op_class(expr, available_ops)  # pytype: disable=wrong-arg-types

  # Simple case without arguments.
  if isinstance(expr, ast.Name):
    return op_class()

  assert isinstance(expr, ast.Call)
  args = [ast.literal_eval(arg) for arg in expr.args]
  kwargs = {kv.arg: ast.literal_eval(kv.value) for kv in expr.keywords}
  if not args:
    return op_class(**kwargs)

  # Translate positional arguments into keyword arguments.
  available_arg_names = [f.name for f in dataclasses.fields(op_class)]
  for i, arg in enumerate(args):
    name = available_arg_names[i]
    if name in kwargs:
      raise ValueError(
          f'Argument {name} to {op_class} given both as positional argument '
          f'(value: {arg}) and keyword argument (value: {kwargs[name]}).')
    kwargs[name] = arg

  return op_class(**kwargs)


def parse(spec: str, available_ops: list[tuple[str, Any]]
          ) -> grain.Transformations:
  """Parses a preprocess spec; a '|' separated list of preprocess ops."""
  available_ops = dict(available_ops)
  if not spec.strip():
    transformations = []
  else:
    transformations = [
        _parse_single_preprocess_op(s, available_ops)
        for s in spec.split('|')
        if s.strip()
    ]

  return transformations

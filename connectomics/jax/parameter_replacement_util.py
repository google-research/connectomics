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
"""A library allowing to manage models with ordered parameter blocks.

This library defines the ordering on atomic parameter blocks (see below for the
definition and examples) and provides functions for replacing a specified number
of final parameter blocks in one model with parameters from another model.


This library assumes that during model evaluation, subcomponents are evaluated
in a sequential order (thus, their parameters can also be ordered). The ordering
must be encoded in the names of parameter keys inside the parameters pytree.

For each node in the pytree, one of the following must be true:
* All child keys are prefixed with the ordering prefix.
* No child keys are prefixed with the ordering prefix.
The ordering prefix consists of the child numer followed by a colon.

Examples of correct nodes:
  "parent_key": {"child":..., "another child": ...}
  "parent_key": {"2:child":..., "2:another child": ...}

Examples of incorrect pytree nodes:
  "parent_key": {"child":..., "1:another child": ...}

A tree node without child ordering is considered as belonging to an atomic model
component. Parameters of such a component will always be replaced together.
This is true even if children of this node contain parameter block ordering.
This mechanism allows treating small submodels (e.g. convolutional layers having
kernel weights + biases) as undivisible.

All atomic pytree nodes can be ordered by running a depth first search on the
pytree with the constraint of visiting non-atomic node children in their
declared order.

For example, let's consider the following tree:
params = {
  "1:a": {
    "1:c": {
      "biases": [1,2,3,4],
      "weights": [0, 0.2]
    },
    "2:d": {
      "biases": [1,2,3,4],
      "weights": [0, 0.2]
    }
  },
  "2:b": {
    "biases": [1,2,3,4],
    "weights": [0, 0.2]
  }
}

This tree defines the following ordering of atomic tree nodes (here represented
by their key in the parent node, which in this example are unique):

["1:c", "2:d", "2:b"]

get_num_atomic_blocks(params)
> 3

is_params_block_atomic(params)
> False
is_params_block_atomic(params["2:b"])
> True

"""
import itertools
from typing import Any, Optional

PyTree = Any


def get_ordered_child_keys(parameters: PyTree) -> list[str]:
  """Returns a list of children node keys sorted in increasing order.

  Args:
    parameters: A pytree of model parameters.

  Returns:
    List of children keys or empty list if there are no children.
  """
  if 'keys' not in dir(parameters):
    # There are no children (we are at the leaf node).
    return []
  keys_to_sort = []

  for layer_key in parameters.keys():
    delimiter_position = layer_key.find(':')
    if delimiter_position == -1:
      continue
    keys_to_sort.append((int(layer_key[:delimiter_position]), layer_key))
  return [x[1] for x in sorted(keys_to_sort)]


def get_num_atomic_blocks(parameters: Any) -> int:
  """Determines the number of parameter blocks which are atomic.

  Args:
    parameters: A pytree of model parameters.

  Returns:
   The number of atomic parameter blocks.
  """
  num_atomic_children = 0
  for child_key in get_ordered_child_keys(parameters):
    num_atomic_children += max(1, get_num_atomic_blocks(parameters[child_key]))
  return num_atomic_children


def is_params_block_atomic(parameters: PyTree) -> bool:
  """Checks whether the passed block of parameters is atomic.

  Args:
    parameters: pytree of model parameters.

  Returns:
    True/False
  """
  return not get_ordered_child_keys(parameters)


def replace_final_parameters(parameters: PyTree, replacements: PyTree,
                             num_to_replace: int) -> int:
  """Replaces the specified number of final model parameters.

  Replaces last parameters when ordered by the
  Args:
    parameters: Pytree containing the original parameters.
    replacements: Pytree of parameters to replace with.
    num_to_replace: The number of atomic parameter blocks to replace.

  Returns:
    The number of replaced parameter blocks.
  """
  num_replaced = 0
  if num_replaced == num_to_replace:
    return num_replaced

  for considered_key in reversed(get_ordered_child_keys(parameters)):
    if is_params_block_atomic(parameters[considered_key]):
      parameters[considered_key] = replacements[considered_key]
      num_replaced += 1
    else:
      num_replaced += replace_final_parameters(parameters[considered_key],
                                               replacements[considered_key],
                                               num_to_replace - num_replaced)
    if num_replaced == num_to_replace:
      return num_replaced
  return num_replaced


class LayerNaming:
  """Generator for model layer names.

  Allows to optionally prefix names with their number. Every time a new name is
  requested, the layer number is increased.
  """

  def __init__(self, should_prefix_names: bool):
    self.should_prefix_names = should_prefix_names
    self.layer_num_iter = itertools.count()

  def get_name(self,
               base_name: str,
               fallback_name: Optional[str] = None) -> str:
    """Generates a layer name.

    If layer numbering is enabled, generates the next layer number and prefixes
    basename with this number. Otherwise, returns base name.

    Args:
      base_name: The base name of the layer.
      fallback_name: Name of the layer to use when name prefixing is disabled.
        If specified, takes precedence over 'base_name'.

    Returns:
      Generated layer name.
    """
    if self.should_prefix_names:
      return f'{next(self.layer_num_iter)}:{base_name}'
    elif fallback_name is not None:
      return fallback_name
    else:
      return base_name

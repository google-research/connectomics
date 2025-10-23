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
"""TensorStore utilities."""

import json
from typing import Any, Mapping

import tensorstore as ts


def write_json(
    to_write: Mapping[str, Any],
    kvstore: ts.KvStore.Spec | str | Mapping[str, Any],
    context: ts.Context | None = None,
) -> Mapping[str, Any]:
  """Write JSON data to parent dir of s0_tstore using TensorStore driver."""
  if context is None:
    context = ts.Context()
  _ = ts.open(
      dict(
          driver='json',
          kvstore=kvstore,
      ), context=context).result().write(to_write).result()
  return to_write


def load_json(
    kvstore: ts.KvStore.Spec | str | Mapping[str, Any],
    context: ts.Context | None = None,
) -> Mapping[str, Any]:
  """Load JSON data using TensorStore driver."""
  if context is None:
    context = ts.Context()
  ds = ts.open(
      dict(
          driver='json',
          kvstore=kvstore,
      ), context=context).result()
  return json.loads(str(ds.read().result()))

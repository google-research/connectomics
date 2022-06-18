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
"""Shim for supporting internal and external file IO."""

# TODO(timblakely): Remove dependency on TF when there's a common API to read
# files internally and externally.
import tensorflow.compat.v2 as tf
import tensorflow.compat.v2.io.gfile as gfile

Copy = gfile.copy
DeleteRecursively = gfile.rmtree
Exists = gfile.exists
Glob = gfile.glob
IsDirectory = gfile.isdir
ListDirectory = gfile.listdir
MakeDirs = gfile.makedirs
MkDir = gfile.mkdir
Open = gfile.GFile
Remove = gfile.remove
Rename = gfile.rename
Stat = gfile.stat
Walk = gfile.walk

NotFoundError = tf.errors.NotFoundError

GFile = gfile.GFile

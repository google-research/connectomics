# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Defines some commonly used model configurations.

Configs were inspired by arxiv.org/abs/2006.10108 (Table 5). We use the same
nomenclature (BERT and WRN). These do not represent configs for the actual
BERT model but for the model trained on top of the BERT embeddings.
"""


BERT_SNGP_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": .95,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 2048,
    "batch_size": 128,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.001,
}

BERT_SNGP_HS_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": .95,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 2048,
    "batch_size": 1024,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.001,
}

BERT_SN_v1 = BERT_SNGP_v1.copy()
BERT_SN_v1["num_inducing"] = 0

BERT_GP_v1 = BERT_SNGP_v1.copy()
BERT_GP_v1["spec_norm_bound"] = 0

BERT_SMALL_SNGP_v1 = {
    "num_hidden": 64,
    "spec_norm_bound": .95,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 256,
    "batch_size": 128,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.001,
}

BERT_SMALL_SN_v1 = BERT_SMALL_SNGP_v1.copy()
BERT_SMALL_SN_v1["num_inducing"] = 0

BERT_SMALL_GP_v1 = BERT_SMALL_SNGP_v1.copy()
BERT_SMALL_GP_v1["spec_norm_bound"] = 0

WRN_SNGP_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": 6.,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 1024,
    "batch_size": 128,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.0001,
}

WRN_SNGP_HS_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": 6.,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 1024,
    "batch_size": 1024,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.0001,
}

WRN_SNGP_VHS_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": 6.,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 1024,
    "batch_size": 2048,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.001,
}

WRN_SNGP_VHS_v2 = {
    "num_hidden": 128,
    "spec_norm_bound": 6.,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 1024,
    "batch_size": 2048,
    "num_layers": 4,
    "use_bn": False,
    "l2_reg": 0.001,
}

WRN_SNGP_HS_v2 = {
    "num_hidden": 512,
    "spec_norm_bound": 6.,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 1024,
    "batch_size": 1024,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.001,
}

WRN_SN_v1 = WRN_SNGP_v1.copy()
WRN_SN_v1["num_inducing"] = 0

WRN_GP_v1 = WRN_SNGP_v1.copy()
WRN_GP_v1["spec_norm_bound"] = 0

WRN_HS_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": 0,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 0,
    "batch_size": 1024,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.001,
}

WRN_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": 0,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 0,
    "batch_size": 128,
    "num_layers": 2,
    "use_bn": False,
    "l2_reg": 0.001,
}

linear_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": 0,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 0,
    "batch_size": 128,
    "num_layers": 0,
    "use_bn": False,
    "l2_reg": 0.0001,
}

linear_HS_v1 = {
    "num_hidden": 128,
    "spec_norm_bound": 0,
    "gp_kernel_scale": 2.,
    "scale_random_features": True,
    "gp_cov_momentum": 0.999,
    "gp_cov_ridge_penalty": 0.001,
    "num_inducing": 0,
    "batch_size": 1024,
    "num_layers": 0,
    "use_bn": False,
    "l2_reg": 0.0001,
}

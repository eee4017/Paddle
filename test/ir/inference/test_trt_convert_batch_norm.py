# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import unittest
from functools import partial
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertBatchNormTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if self.dims == 4:
                if attrs[0]['data_layout'] == "NCHW":
                    return np.ones([batch, 3, 24, 24]).astype(np.float32)
                elif attrs[0]['data_layout'] == "NHWC":
                    return np.ones([batch, 24, 24, 3]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 3]).astype(np.float32)

        def generate_bias(attrs: List[Dict[str, Any]], batch):
            return np.full((3), 0.9).astype("float32")

        def generate_mean(attrs: List[Dict[str, Any]], batch):
            return np.full((3), 0.9).astype("float32")

        def generate_scale(attrs: List[Dict[str, Any]], batch):
            return np.full((3), 1.1).astype("float32")

        def generate_variance(attrs: List[Dict[str, Any]], batch):
            return np.full((3), 1.2).astype("float32")

        def generate_MomentumTensor(attrs: List[Dict[str, Any]], batch):
            return np.full((3), 0.9).astype("float32")

        dims_list = [2, 3, 4]
        num_input_list = [0, 1]
        batch_list = [1, 4]
        epsilon_list = [1e-06, 1e-05, 0.0001]
        data_layout_list = ['NCHW']
        momentum_list = [0.9, 0.8]
        grid = [
            dims_list,
            num_input_list,
            batch_list,
            epsilon_list,
            data_layout_list,
            momentum_list,
        ]
        for (
            dims,
            num_input,
            batch,
            epsilon,
            data_layout,
            momentum,
        ) in itertools.product(*grid):
            self.num_input = num_input
            self.dims = dims
            dics = [
                {
                    'epsilon': epsilon,
                    'data_layout': data_layout,
                    'momentum': momentum,
                    'is_test': True,
                    'trainable_statistics': False,
                },
                {},
            ]
            dics_intput = [
                {
                    'X': ['batch_norm_input'],
                    'Bias': ['Bias'],
                    'Mean': ['Mean'],
                    'Scale': ['Scale'],
                    'Variance': ['Variance'],
                    'MomentumTensor': ['MomentumTensor'],
                },
                {
                    'X': ['batch_norm_input'],
                    'Bias': ['Bias'],
                    'Mean': ['Mean'],
                    'Scale': ['Scale'],
                    'Variance': ['Variance'],
                },
            ]
            dics_intputs = [
                {
                    'Bias': TensorConfig(
                        data_gen=partial(generate_bias, dics, batch)
                    ),
                    'Mean': TensorConfig(
                        data_gen=partial(generate_mean, dics, batch)
                    ),
                    'Scale': TensorConfig(
                        data_gen=partial(generate_scale, dics, batch)
                    ),
                    'Variance': TensorConfig(
                        data_gen=partial(generate_variance, dics, batch)
                    ),
                    'MomentumTensor': TensorConfig(
                        data_gen=partial(generate_MomentumTensor, dics, batch)
                    ),
                },
                {
                    'Bias': TensorConfig(
                        data_gen=partial(generate_bias, dics, batch)
                    ),
                    'Mean': TensorConfig(
                        data_gen=partial(generate_mean, dics, batch)
                    ),
                    'Scale': TensorConfig(
                        data_gen=partial(generate_scale, dics, batch)
                    ),
                    'Variance': TensorConfig(
                        data_gen=partial(generate_variance, dics, batch)
                    ),
                },
            ]
            ops_config = [
                {
                    'op_type': 'batch_norm',
                    'op_inputs': dics_intput[num_input],
                    'op_outputs': {
                        'Y': ['batch_norm_out'],
                        'MeanOut': ['Mean'],
                        'VarianceOut': ['Variance'],
                        'SavedMean': ['SavedMean'],
                        'SavedVariance': ['SavedVariance'],
                    },
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights=dics_intputs[num_input],
                inputs={
                    'batch_norm_input': TensorConfig(
                        data_gen=lambda: generate_input1(dics, batch)
                    )
                },
                outputs=['batch_norm_out'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                if attrs[0]['data_layout'] == "NCHW":
                    self.dynamic_shape.min_input_shape = {
                        "batch_norm_input": [1, 3, 12, 12]
                    }
                    self.dynamic_shape.max_input_shape = {
                        "batch_norm_input": [4, 3, 24, 24]
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "batch_norm_input": [1, 3, 24, 24]
                    }
                elif attrs[0]['data_layout'] == "NHWC":
                    self.dynamic_shape.min_input_shape = {
                        "batch_norm_input": [1, 12, 12, 3]
                    }
                    self.dynamic_shape.max_input_shape = {
                        "batch_norm_input": [4, 24, 24, 3]
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "batch_norm_input": [1, 24, 24, 3]
                    }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "batch_norm_input": [1, 3, 12]
                }
                self.dynamic_shape.max_input_shape = {
                    "batch_norm_input": [4, 3, 24]
                }
                self.dynamic_shape.opt_input_shape = {
                    "batch_norm_input": [1, 3, 24]
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "batch_norm_input": [1, 3]
                }
                self.dynamic_shape.max_input_shape = {
                    "batch_norm_input": [4, 3]
                }
                self.dynamic_shape.opt_input_shape = {
                    "batch_norm_input": [1, 3]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # for static_shape
        clear_dynamic_shape()
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (
                self.create_inference_config(),
                generate_trt_nodes_num(attrs, False),
                1e-05,
            )
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (
                self.create_inference_config(),
                generate_trt_nodes_num(attrs, False),
                (1e-03, 1e-03),
            )
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (
                self.create_inference_config(),
                generate_trt_nodes_num(attrs, True),
                1e-05,
            )
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (
                self.create_inference_config(),
                generate_trt_nodes_num(attrs, True),
                (1e-03, 1e-03),
            )

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(program_config.weights) == 5:
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_SUPPORT,
            "INPUT MomentumTensor NOT SUPPORT",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()

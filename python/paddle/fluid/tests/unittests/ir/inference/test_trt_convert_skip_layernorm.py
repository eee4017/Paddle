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
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertSkipLayernormTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # The input dimension should be less than or equal to the set axis.
        if attrs[0]['begin_norm_axis'] >= 0:
            if (
                len(inputs['skip_layernorm_inputX_data'].shape)
                <= attrs[0]['begin_norm_axis']
            ):
                return False

        # 2D input is not supported.
        if self.dims == 2:
            return False
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32]

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if self.dims == 4:
                return np.ones([batch, 6, 128, 768]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 128, 768]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 768]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], batch):
            if self.dims == 4:
                return np.ones([batch, 6, 128, 768]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 128, 768]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 768]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.random.random([768]).astype(np.float32)

        def generate_weight2(attrs: List[Dict[str, Any]]):
            return np.random.random([768]).astype(np.float32)

        dims_list = [2, 3, 4]
        batch_list = [1, 2, 4]
        epsilon_list = [1e-05]
        begin_norm_axis_list = [0, 1, 2, -1]
        enable_int8_list = [False, True]
        grid = [
            dims_list,
            batch_list,
            epsilon_list,
            begin_norm_axis_list,
            enable_int8_list,
        ]
        for (
            dims,
            batch,
            epsilon,
            begin_norm_axis,
            enable_int8,
        ) in itertools.product(*grid):
            self.dims = dims
            dics = [
                {
                    'epsilon': epsilon,
                    'begin_norm_axis': begin_norm_axis,
                    'enable_int8': enable_int8,
                },
                {},
            ]
            ops_config = [
                {
                    'op_type': 'skip_layernorm',
                    'op_inputs': {
                        'X': ['skip_layernorm_inputX_data'],
                        'Y': ['skip_layernorm_inputY_data'],
                        'Bias': ['Bias'],
                        'Scale': ['Scale'],
                    },
                    'op_outputs': {'Out': ['skip_layernorm_out']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={
                    'Bias': TensorConfig(
                        data_gen=partial(generate_weight1, dics)
                    ),
                    'Scale': TensorConfig(
                        data_gen=partial(generate_weight2, dics)
                    ),
                },
                inputs={
                    'skip_layernorm_inputX_data': TensorConfig(
                        data_gen=lambda: generate_input1(dics, batch)
                    ),
                    'skip_layernorm_inputY_data': TensorConfig(
                        data_gen=lambda: generate_input2(dics, batch)
                    ),
                },
                outputs=['skip_layernorm_out'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "skip_layernorm_inputX_data": [1, 6, 128, 768],
                    "skip_layernorm_inputY_data": [1, 6, 128, 768],
                    "Bias": [768],
                    "Scale": [768],
                }
                self.dynamic_shape.max_input_shape = {
                    "skip_layernorm_inputX_data": [4, 6, 768, 3072],
                    "skip_layernorm_inputY_data": [4, 6, 768, 3072],
                    "Bias": [3072],
                    "Scale": [3072],
                }
                self.dynamic_shape.opt_input_shape = {
                    "skip_layernorm_inputX_data": [2, 6, 128, 768],
                    "skip_layernorm_inputY_data": [2, 6, 128, 768],
                    "Bias": [768],
                    "Scale": [768],
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "skip_layernorm_inputX_data": [1, 128, 768],
                    "skip_layernorm_inputY_data": [1, 128, 768],
                    "Bias": [768],
                    "Scale": [768],
                }
                self.dynamic_shape.max_input_shape = {
                    "skip_layernorm_inputX_data": [4, 768, 3072],
                    "skip_layernorm_inputY_data": [4, 768, 3072],
                    "Bias": [3072],
                    "Scale": [3072],
                }
                self.dynamic_shape.opt_input_shape = {
                    "skip_layernorm_inputX_data": [2, 128, 768],
                    "skip_layernorm_inputY_data": [2, 128, 768],
                    "Bias": [768],
                    "Scale": [768],
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "skip_layernorm_inputX_data": [1, 768],
                    "skip_layernorm_inputY_data": [1, 768],
                    "Bias": [768],
                    "Scale": [768],
                }
                self.dynamic_shape.max_input_shape = {
                    "skip_layernorm_inputX_data": [4, 3072],
                    "skip_layernorm_inputY_data": [4, 3072],
                    "Bias": [3072],
                    "Scale": [3072],
                }
                self.dynamic_shape.opt_input_shape = {
                    "skip_layernorm_inputX_data": [2, 768],
                    "skip_layernorm_inputY_data": [2, 768],
                    "Bias": [768],
                    "Scale": [768],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape == True:
                return 1, 3
            else:
                return 0, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # # for static_shape
        # clear_dynamic_shape()

        # self.trt_param.precision = paddle_infer.PrecisionType.Float32
        # yield self.create_inference_config(), generate_trt_nodes_num(
        #     attrs, False), 1e-5
        # self.trt_param.precision = paddle_infer.PrecisionType.Half
        # yield self.create_inference_config(), generate_trt_nodes_num(
        #     attrs, False), (1e-3, 1e-3)

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
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()

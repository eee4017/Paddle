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
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertLeakyReluTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16, np.int8]

    def sample_program_configs(self):
        def generate_input1(shape):
            return np.random.random(shape).astype(np.float32)

        batch_list = [1, 2]
        shape_list = [[64], [32, 64], [8, 32, 32]]
        alpha_list = [0.02, 1.0, 100.0, -1.0, 0.0]
        grid = [batch_list, shape_list, alpha_list]
        for batch, shape, alpha in itertools.product(*grid):
            shape = [batch, *shape]
            self.input_dim = len(shape)
            dics = [{'alpha': alpha}]
            ops_config = [
                {
                    'op_type': 'leaky_relu',
                    'op_inputs': {'X': ['input_data']},
                    'op_outputs': {'Out': ['y_data']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    'input_data': TensorConfig(
                        data_gen=lambda: generate_input1(shape)
                    )
                },
                outputs=['y_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.input_dim == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 8]}
                self.dynamic_shape.max_input_shape = {"input_data": [64, 128]}
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 16]}
            elif self.input_dim == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 8, 8]}
                self.dynamic_shape.max_input_shape = {
                    "input_data": [64, 128, 256]
                }
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 16, 64]}
            elif self.input_dim == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 8, 8, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [64, 64, 128, 128]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 16, 64, 32]
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
        if program_config.get_input_type() == np.int8:
            self.trt_param.precision = paddle_infer.PrecisionType.Int8
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
        if program_config.get_input_type() == np.int8:
            self.trt_param.precision = paddle_infer.PrecisionType.Int8
            yield (
                self.create_inference_config(),
                generate_trt_nodes_num(attrs, True),
                (1e-03, 1e-03),
            )

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()

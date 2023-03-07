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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import itertools
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest


class TrtConvertMatmulTest_static(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        batch_list = [1, 4]
        trans_x_list = [True, False]
        trans_y_list = [True, False]
        alpha_list = [0.3, 1.0]
        grid = [batch_list, trans_x_list, trans_y_list, alpha_list]
        for batch, trans_x, trans_y, alpha in itertools.product(*grid):
            if trans_x and trans_y:
                input1_shape = [batch, 6, 11]
                input2_shape = [batch, 32, 6]
            if trans_x and (not trans_y):
                input1_shape = [batch, 6, 11]
                input2_shape = [batch, 6, 32]
            if not trans_x and trans_y:
                input1_shape = [batch, 32, 6]
                input2_shape = [batch, 11, 6]
            if not trans_x and (not trans_y):
                input1_shape = [batch, 32, 6]
                input2_shape = [batch, 6, 11]
            dics = [
                {
                    'transpose_X': trans_x,
                    'transpose_Y': trans_y,
                    'alpha': alpha,
                    'fused_reshape_X': [],
                    'fused_reshape_Y': [],
                    'fused_transpose_X': [],
                    'fused_transpose_Y': [],
                    'fused_reshape_Out': [],
                    'fused_transpose_Out': [],
                }
            ]
            ops_config = [
                {
                    'op_type': 'matmul',
                    'op_inputs': {'X': ['input1_data'], 'Y': ['input2_data']},
                    'op_outputs': {'Out': ['output_data']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    'input1_data': TensorConfig(
                        data_gen=lambda: generate_input(input1_shape)
                    ),
                    'input2_data': TensorConfig(
                        data_gen=lambda: generate_input(input2_shape)
                    ),
                },
                outputs=['output_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            pass

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        # for static_shape
        clear_dynamic_shape()
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (self.create_inference_config(), (1, 3), 1e-05)
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (1, 3), 1e-03)

    def test(self):
        self.run_test()


class TrtConvertMatmulTest_dynamic(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        trans_x_list = [True]
        trans_y_list = [True]
        alpha_list = [0.3, 1.0]
        grid = [trans_x_list, trans_y_list, alpha_list]
        for trans_x, trans_y, alpha in itertools.product(*grid):
            dics = [
                {
                    'transpose_X': trans_x,
                    'transpose_Y': trans_y,
                    'alpha': alpha,
                    'fused_reshape_X': [],
                    'fused_reshape_Y': [],
                    'fused_transpose_X': [],
                    'fused_transpose_Y': [],
                    'fused_reshape_Out': [],
                    'fused_transpose_Out': [],
                }
            ]
            ops_config = [
                {
                    'op_type': 'matmul',
                    'op_inputs': {'X': ['input1_data'], 'Y': ['input2_data']},
                    'op_outputs': {'Out': ['output_data']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    'input1_data': TensorConfig(
                        data_gen=lambda: generate_input(input1_shape)
                    ),
                    'input2_data': TensorConfig(
                        data_gen=lambda: generate_input(input2_shape)
                    ),
                },
                outputs=['output_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input1_data": [1, 4, 4],
                "input2_data": [1, 4, 4],
            }
            self.dynamic_shape.max_input_shape = {
                "input1_data": [16, 4, 4],
                "input2_data": [16, 4, 4],
            }
            self.dynamic_shape.opt_input_shape = {
                "input1_data": [8, 4, 4],
                "input2_data": [8, 4, 4],
            }

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (self.create_inference_config(), (1, 3), 1e-05)
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (1, 3), 1e-03)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()

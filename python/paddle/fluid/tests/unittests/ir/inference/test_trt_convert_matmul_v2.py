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
import os


class TrtConvertMatmulTest_dynamic(TrtLayerAutoScanTest):
    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        batch_list = [10, 11, 12, 13, 14, 15]
        trans_x_list = [False]
        trans_y_list = [False]
        grid = [batch_list, trans_x_list, trans_y_list]
        for batch, trans_x, trans_y in itertools.product(*grid):
            input1_shape = [batch, 64, 350, 75]
            input2_shape = [75, 25]
            dics = [{'trans_x': trans_x, 'trans_y': trans_y}]
            ops_config = [
                {
                    'op_type': 'matmul_v2',
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
                "input1_data": [10, 64, 350, 75],
                "input2_data": [75, 25],
            }
            self.dynamic_shape.max_input_shape = {
                "input1_data": [100, 64, 350, 75],
                "input2_data": [75, 25],
            }
            self.dynamic_shape.opt_input_shape = {
                "input1_data": [15, 64, 350, 75],
                "input2_data": [75, 25],
            }

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # The output has little diff between gpu and trt in CI-Windows-Inference
        tol_fp32 = 1e-5
        # The half threshold is set 4e-3 because matmul with large size has larger difference
        tol_half = 4e-3
        if os.name == 'nt':
            tol_fp32 = 1e-3
            tol_half = 4e-3
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (self.create_inference_config(), (1, 3), (tol_fp32, tol_fp32))
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (1, 3), (tol_half, tol_half))

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertMatmulTest_dynamic2(TrtLayerAutoScanTest):
    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        batch_list = [10, 11, 12, 13, 14, 15]
        trans_x_list = [False]
        trans_y_list = [False]
        grid = [batch_list, trans_x_list, trans_y_list]
        for batch, trans_x, trans_y in itertools.product(*grid):
            input1_shape = [60, 40]
            input2_shape = [batch, 40, 90]
            dics = [{'trans_x': trans_x, 'trans_y': trans_y}]
            ops_config = [
                {
                    'op_type': 'matmul_v2',
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
                "input1_data": [60, 40],
                "input2_data": [10, 40, 90],
            }
            self.dynamic_shape.max_input_shape = {
                "input1_data": [60, 40],
                "input2_data": [20, 40, 90],
            }
            self.dynamic_shape.opt_input_shape = {
                "input1_data": [60, 40],
                "input2_data": [15, 40, 90],
            }

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # The output has little diff between gpu and trt in CI-Windows-Inference
        tol_fp32 = 1e-5
        tol_half = 4e-3
        if os.name == 'nt':
            tol_fp32 = 1e-3
            tol_half = 4e-3
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (self.create_inference_config(), (1, 3), (tol_fp32, tol_fp32))
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (1, 3), (tol_half, tol_half))

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertMatmulTest_dynamic3(TrtLayerAutoScanTest):
    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        # case0: mat * vec
        # case1: vec * mat
        # case2: vec * vec
        case_list = [0, 1, 2]
        batch_list = range(20, 23)
        trans_x_list = [False, True]
        trans_y_list = [False, True]
        grid = [case_list, batch_list, trans_x_list, trans_y_list]
        for case, batch, trans_x, trans_y in itertools.product(*grid):
            self.case = case
            input1_shape = []
            input2_shape = []
            if case == 0:
                input1_shape = [batch, 50]
                input2_shape = [50]
            elif case == 1:
                input1_shape = [50]
                input2_shape = [50, batch]
            elif case == 2:
                input1_shape = [50]
                input2_shape = [50]
            if case == 0 or case == 1:
                dics = [{'trans_x': False, 'trans_y': False}]
            elif case == 2:
                dics = [{'trans_x': trans_x, 'trans_y': trans_y}]
            ops_config = [
                {
                    'op_type': 'matmul_v2',
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
        def generate_dynamic_shape():
            if self.case == 0:
                self.dynamic_shape.min_input_shape = {
                    "input1_data": [20, 50],
                    "input2_data": [50],
                }
                self.dynamic_shape.max_input_shape = {
                    "input1_data": [30, 50],
                    "input2_data": [50],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input1_data": [25, 50],
                    "input2_data": [50],
                }
            elif self.case == 1:
                self.dynamic_shape.min_input_shape = {
                    "input2_data": [50, 20],
                    "input1_data": [50],
                }
                self.dynamic_shape.max_input_shape = {
                    "input2_data": [50, 30],
                    "input1_data": [50],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input2_data": [50, 25],
                    "input1_data": [50],
                }
            elif self.case == 2:
                self.dynamic_shape.min_input_shape = {
                    "input2_data": [30],
                    "input1_data": [50],
                }
                self.dynamic_shape.max_input_shape = {
                    "input2_data": [50],
                    "input1_data": [50],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input2_data": [50],
                    "input1_data": [50],
                }

        generate_dynamic_shape()
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (self.create_inference_config(), (1, 3), (1e-05, 1e-05))
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (1, 3), (4e-03, 4e-03))

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()

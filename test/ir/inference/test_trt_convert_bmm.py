# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertBmmTest_dynamic(TrtLayerAutoScanTest):
    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        batch_list = [10, 11, 12, 13, 14, 15]
        for batch in batch_list:
            input1_shape = [batch, 350, 75]
            input2_shape = [batch, 75, 25]
            dics = [{}]
            ops_config = [
                {
                    'op_type': 'bmm',
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
                "input1_data": [10, 350, 75],
                "input2_data": [10, 75, 25],
            }
            self.dynamic_shape.max_input_shape = {
                "input1_data": [100, 350, 75],
                "input2_data": [100, 75, 25],
            }
            self.dynamic_shape.opt_input_shape = {
                "input1_data": [15, 350, 75],
                "input2_data": [15, 75, 25],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                return 1, 3
            else:
                return 0, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

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
                1e-03,
            )
        # The output has little diff between gpu and trt in CI-Windows-Inference
        tol_fp32 = 1e-04
        tol_half = 5e-03
        if os.name == 'nt':
            tol_fp32 = 1e-02
            tol_half = 1e-02
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (
                self.create_inference_config(),
                generate_trt_nodes_num(attrs, True),
                (tol_fp32, tol_fp32),
            )
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (
                self.create_inference_config(),
                generate_trt_nodes_num(attrs, True),
                (tol_half, tol_half),
            )

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()

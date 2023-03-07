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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import itertools
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
from paddle.framework import convert_np_dtype_to_dtype_


class TrtConvertCastTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if attrs[0]['in_dtype'] == 0:
            return False
        if attrs[0]['in_dtype'] in [4, 5] and attrs[0]['out_dtype'] == 4:
            return False
        if attrs[0]['in_dtype'] not in [2, 4, 5] or attrs[0][
            'out_dtype'
        ] not in [2, 4, 5]:
            return False
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input(dtype):
            return np.ones([1, 3, 64, 64]).astype(dtype)

        in_dtype_list = [np.bool_, np.int32, np.float32, np.float64]
        out_dtype_list = [np.bool_, np.int32, np.float32, np.float64]
        grid = [in_dtype_list, out_dtype_list]
        for in_dtype, out_dtype in itertools.product(*grid):
            dics = [
                {
                    'in_dtype': convert_np_dtype_to_dtype_(in_dtype),
                    'out_dtype': convert_np_dtype_to_dtype_(out_dtype),
                },
                {
                    'in_dtype': convert_np_dtype_to_dtype_(out_dtype),
                    'out_dtype': convert_np_dtype_to_dtype_(in_dtype),
                },
            ]
            ops_config = [
                {
                    'op_type': 'cast',
                    'op_inputs': {'X': ['input_data']},
                    'op_outputs': {'Out': ['cast_output_data0']},
                    'op_attrs': dics[0],
                    'op_outputs_dtype': {'cast_output_data0': out_dtype},
                },
                {
                    'op_type': 'cast',
                    'op_inputs': {'X': ['cast_output_data0']},
                    'op_outputs': {'Out': ['cast_output_data1']},
                    'op_attrs': dics[1],
                    'op_outputs_dtype': {'cast_output_data1': in_dtype},
                },
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    'input_data': TensorConfig(
                        data_gen=lambda: generate_input(in_dtype)
                    )
                },
                outputs=['cast_output_data1'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}

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
                1e-02,
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
                1e-02,
            )

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()

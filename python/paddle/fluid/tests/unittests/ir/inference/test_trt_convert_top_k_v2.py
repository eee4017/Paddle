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
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertActivationTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if len(inputs['input_data'].shape) <= attrs[0]['axis']:
            return False
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(dims, batch, attrs: List[Dict[str, Any]]):
            data_shape = [3, 32, 32]
            if dims in [1, 2, 3]:
                input_dim = data_shape[:dims]
            else:
                input_dim = [batch, 8, 8, 8]
            return (
                np.random.choice(
                    np.arange(-2048, 2048), input_dim, replace=False
                )
                .astype(np.float16)
                .astype(np.float32)
            )

        dims_list = [1, 2, 3, 4]
        batch_list = [1, 4]
        k_list = [1, 3]
        axis_list = [-1, 1, 2, 3]
        largest_list = [True, False]
        sort_list = [True, False]
        grid = [
            dims_list,
            batch_list,
            k_list,
            axis_list,
            largest_list,
            sort_list,
        ]
        for dims, batch, k, axis, largest, sort in itertools.product(*grid):
            self.dims = dims
            self.sort = sort
            dics = [{'k': k, 'axis': axis, 'largest': largest, 'sorted': sort}]
            ops_config = [
                {
                    'op_type': 'top_k_v2',
                    'op_inputs': {'X': ['input_data']},
                    'op_outputs': {
                        'Out': ['output_data'],
                        'Indices': ['indices_data'],
                    },
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    'input_data': TensorConfig(
                        data_gen=lambda: generate_input1(dims, batch, dics)
                    )
                },
                outputs=['output_data', 'indices_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [1]}
                self.dynamic_shape.max_input_shape = {"input_data": [64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [32]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 1]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 10]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 1]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 10, 10]}
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 8, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 32, 32, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 32, 32]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if self.dims == 1:
                return 0, 4
            if self.sort == False:
                return 0, 4
            return 1, 3

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
                1e-03,
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
                1e-03,
            )

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()

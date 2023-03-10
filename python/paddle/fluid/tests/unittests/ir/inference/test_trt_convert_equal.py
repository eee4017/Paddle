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


class TrtConvertElementwiseTest_one_input_corner_case(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if attrs[0]['axis'] == 0:
            return false
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8415:
            return False
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.choice(3, shape).astype(np.float32)

        batch_list = [1, 2, 4]
        shape_list = [[1], [1, 32], [1, 32, 16]]
        grid = [batch_list, shape_list]
        for batch, shape in itertools.product(*grid):
            shape = [batch, *shape]
            axis = 1
            self.dims = len(shape)
            dics = [{'axis': axis}, {'in_dtype': 0, 'out_dtype': 5}]
            ops_config = [
                {
                    'op_type': 'equal',
                    'op_inputs': {'X': ['input_data1'], 'Y': ['input_data2']},
                    'op_outputs': {'Out': ['compare_output_data']},
                    'op_attrs': dics[0],
                    'op_output_dtype': {'compare_output_data': np.bool_},
                },
                {
                    'op_type': 'cast',
                    'op_inputs': {'X': ['compare_output_data']},
                    'op_outputs': {'Out': ['output_data']},
                    'op_attrs': dics[1],
                },
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    'input_data1': TensorConfig(
                        data_gen=lambda: generate_input(shape)
                    ),
                    'input_data2': TensorConfig(
                        data_gen=lambda: generate_input(shape)
                    ),
                },
                outputs=['output_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            # The input.dims[1] must be equal to the weight's length.
            min_input_shape = [1, 1, 2, 2]
            max_input_shape = [4, 1, 32, 16]
            opt_input_shape = [2, 1, 16, 8]
            self.dynamic_shape.min_input_shape = {
                "input_data1": min_input_shape[: self.dims],
                "input_data2": min_input_shape[: self.dims],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": max_input_shape[: self.dims],
                "input_data2": max_input_shape[: self.dims],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": opt_input_shape[: self.dims],
                "input_data2": opt_input_shape[: self.dims],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # We only test dynamic shape because TRT does not support BOOL output with static shape
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
        self.trt_param.workspace_size = 1 << 20
        self.run_test()


if __name__ == "__main__":
    unittest.main()

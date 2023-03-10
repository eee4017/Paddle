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

# The elementwise_pow half kernel may results in infinity or large output difference in comparison to float kernel
# Hence, we generate the base in [0.5, 1.5) instead of [0.0, 1.0) and round the exponential part
def generate_input(shape, is_pow=False):
    np_data = np.random.random(shape).astype(np.float32)
    if is_pow:
        return np_data + 0.5
    return np_data


def generate_weight(shape, is_pow=False, is_div=False):
    if is_pow:
        # notice that the base should not be negative or it might cause undefined behavior (negative ^ float may be a complex number)
        np_data = np.random.random(shape).astype(np.float32)
        return np_data.round(1)
    if is_div:
        np_data = np.random.random(shape).astype(np.float32)
        return np_data + 1.0
    else:
        np_data = np.random.randn(*shape).astype(np.float32)
        return np_data


# This is the special test case with weight including batch dimension
# I don't want to mess up the code written by others, so I wrote a class specifically
class TrtConvertElementwiseTest_one_input_special_case0(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        batch_list = [1, 4]
        shape_list = [[32, 4, 4]]
        op_type_list = [
            'elementwise_add',
            'elementwise_mul',
            'elementwise_sub',
            'elementwise_div',
            'elementwise_pow',
            'elementwise_min',
            'elementwise_max',
        ]
        grid = [batch_list, shape_list, op_type_list]
        for batch, shape, op_type in itertools.product(*grid):
            shape = [batch, *shape]
            is_pow = op_type == 'elementwise_pow'
            axis = -1
            self.dims = len(shape)
            dics = [{'axis': axis}]
            ops_config = [
                {
                    'op_type': op_type,
                    'op_inputs': {'X': ['input_data'], 'Y': ['weight']},
                    'op_outputs': {'Out': ['output_data']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={
                    'weight': TensorConfig(
                        data_gen=partial(
                            generate_weight, (1, 32, 1, 1), is_pow=is_pow
                        )
                    )
                },
                inputs={
                    'input_data': TensorConfig(
                        data_gen=lambda: generate_input(shape, is_pow=is_pow)
                    )
                },
                outputs=['output_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            # The input.dims[1] must be equal to the weight's length.
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 32, 4, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 32, 32, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [4, 32, 8, 8]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
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
                (1e-05, 1e-05),
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
                (1e-05, 1e-05),
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


# This is the special test case
class TrtConvertElementwiseTest_one_input_special_case1(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):
        shape_list = [[32]]
        op_type_list = [
            'elementwise_add',
            'elementwise_mul',
            'elementwise_sub',
            'elementwise_div',
            'elementwise_pow',
            'elementwise_min',
            'elementwise_max',
        ]
        grid = [shape_list, op_type_list]
        for shape, op_type in itertools.product(*grid):
            is_pow = op_type == 'elementwise_pow'
            axis = -1
            self.dims = len(shape)
            dics = [{'axis': axis}]
            ops_config = [
                {
                    'op_type': op_type,
                    'op_inputs': {'X': ['input_data'], 'Y': ['weight']},
                    'op_outputs': {'Out': ['output_data']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={
                    'weight': TensorConfig(
                        data_gen=partial(generate_weight, (1,), is_pow=is_pow)
                    )
                },
                inputs={
                    'input_data': TensorConfig(
                        data_gen=lambda: generate_input(shape, is_pow=is_pow)
                    )
                },
                outputs=['output_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [32]}
            self.dynamic_shape.max_input_shape = {"input_data": [64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [32]}

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape:
                return 0, 3
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
                (1e-05, 1e-05),
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
                (1e-05, 1e-05),
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


class TrtConvertElementwiseTest_one_input(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):

        batch_list = [1, 4]
        shape_list = [[32], [32, 8], [32, 8, 4]]
        op_type_list = [
            'elementwise_add',
            'elementwise_mul',
            'elementwise_sub',
            'elementwise_div',
            'elementwise_pow',
            'elementwise_min',
            'elementwise_max',
        ]
        grid = [batch_list, shape_list, op_type_list]
        for batch, shape, op_type in itertools.product(*grid):
            is_pow = op_type == 'elementwise_pow'
            is_div = op_type == 'elementwise_div'
            shape = [batch, *shape]
            axis = -1 if len(shape) == 1 else 1
            self.dims = len(shape)
            dics = [{'axis': axis}]
            ops_config = [
                {
                    'op_type': op_type,
                    'op_inputs': {'X': ['input_data'], 'Y': ['weight']},
                    'op_outputs': {'Out': ['output_data']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={
                    'weight': TensorConfig(
                        data_gen=partial(
                            generate_weight, (32,), is_pow=is_pow, is_div=is_div
                        )
                    )
                },
                inputs={
                    'input_data': TensorConfig(
                        data_gen=lambda: generate_input(shape, is_pow=is_pow)
                    )
                },
                outputs=['output_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            # The input.dims[1] must be equal to the weight's length.
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [4]}
                self.dynamic_shape.max_input_shape = {"input_data": [32]}
                self.dynamic_shape.opt_input_shape = {"input_data": [16]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 32]}
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 32]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32, 4]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 32, 32]}
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 32, 32]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 32, 4, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 32, 32, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [4, 32, 16, 32]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if self.dims == 1 and not dynamic_shape:
                return 0, 3
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
                (1e-05, 1e-05),
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
                (1e-05, 1e-05),
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


class TrtConvertElementwiseTest_two_input_without_broadcast(
    TrtLayerAutoScanTest
):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):

        shape_list = [[4], [4, 32], [2, 32, 16], [1, 8, 16, 32]]
        op_type_list = [
            'elementwise_add',
            'elementwise_mul',
            'elementwise_sub',
            'elementwise_div',
            'elementwise_pow',
            'elementwise_min',
            'elementwise_max',
        ]
        axis_list = [0, -1]
        grid = [shape_list, op_type_list, axis_list]
        for shape, op_type, axis in itertools.product(*grid):
            is_pow = op_type == 'elementwise_pow'
            self.dims = len(shape)
            dics = [{'axis': axis}]
            ops_config = [
                {
                    'op_type': op_type,
                    'op_inputs': {'X': ['input_data1'], 'Y': ['input_data2']},
                    'op_outputs': {'Out': ['output_data']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    'input_data1': TensorConfig(
                        data_gen=lambda: generate_input(shape, is_pow=is_pow)
                    ),
                    'input_data2': TensorConfig(
                        data_gen=lambda: generate_input(shape, is_pow=is_pow)
                    ),
                },
                outputs=['output_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1],
                    "input_data2": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [128],
                    "input_data2": [128],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [32],
                    "input_data2": [32],
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 4],
                    "input_data2": [1, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [128, 256],
                    "input_data2": [128, 256],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [32, 64],
                    "input_data2": [32, 64],
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 4, 4],
                    "input_data2": [1, 4, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [128, 128, 256],
                    "input_data2": [128, 128, 256],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 32, 16],
                    "input_data2": [2, 32, 16],
                }
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 4, 4, 4],
                    "input_data2": [1, 4, 4, 4],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [8, 128, 64, 128],
                    "input_data2": [8, 128, 64, 128],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 64, 32, 32],
                    "input_data2": [2, 64, 32, 32],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if self.dims == 1 and not dynamic_shape:
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
                (1e-05, 1e-05),
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
            yield (self.create_inference_config(), (1, 3), (1e-05, 1e-05))
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (1, 3), (1e-03, 1e-03))

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertElementwiseTest_two_input_with_broadcast(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        if len(inputs['input_data1'].shape) != len(inputs['input_data2'].shape):
            return False

        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):

        input1_shape_list = [[4, 32], [2, 4, 32], [4, 2, 4, 32]]
        input2_shape1_list = [[32], [4, 32], [2, 4, 32]]
        input2_shape2_list = [[4, 1], [2, 4, 1], [4, 2, 4, 1]]
        input2_shape3_list = [[32], [2, 1, 1], [4, 2, 1, 32]]
        input2_shape4_list = [[32], [4, 32], [4, 1, 4, 32]]
        input2_shape5_list = [[32], [2, 1, 32], [4, 1, 1, 32]]
        input2_shape6_list = [[1, 32], [1, 32], [1, 1, 1, 32]]
        input2_shape_list = [
            input2_shape1_list,
            input2_shape2_list,
            input2_shape3_list,
            input2_shape4_list,
            input2_shape5_list,
            input2_shape6_list,
        ]
        axis1_list = [[-1], [1, -1], [1, -1]]
        axis2_list = [[-1], [0], [0]]
        axis3_list = [[-1], [0], [0]]
        axis4_list = [[-1], [-1], [0]]
        axis5_list = [[-1, 1], [-1, 0], [-1, 0]]
        axis6_list = [[-1, 0], [-1, 1], [-1, 0]]
        axis_list = [
            axis1_list,
            axis2_list,
            axis3_list,
            axis4_list,
            axis5_list,
            axis6_list,
        ]

        i_list = range(3)
        j_list = range(6)
        op_type_list = [
            'elementwise_add',
            'elementwise_mul',
            'elementwise_sub',
            'elementwise_div',
            'elementwise_pow',
            'elementwise_min',
            'elementwise_max',
        ]
        k_list = range(2)
        grid = [i_list, j_list, op_type_list, k_list]
        for i, j, op_type, k in itertools.product(*grid):
            if k < len(axis_list[j][i]):
                axis = axis_list[j][i][k]
                input1_shape = input1_shape_list[i]
                input2_shape = input2_shape_list[j][i]
                is_pow = op_type == 'elementwise_pow'
                self.shape1 = input1_shape
                self.shape2 = input2_shape
                dics = [{'axis': axis}]
                ops_config = [
                    {
                        'op_type': op_type,
                        'op_inputs': {
                            'X': ['input_data1'],
                            'Y': ['input_data2'],
                        },
                        'op_outputs': {'Out': ['output_data']},
                        'op_attrs': dics[0],
                    }
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        'input_data1': TensorConfig(
                            data_gen=lambda: generate_input(
                                input1_shape, is_pow=is_pow
                            )
                        ),
                        'input_data2': TensorConfig(
                            data_gen=lambda: generate_input(
                                input2_shape, is_pow=is_pow
                            )
                        ),
                    },
                    outputs=['output_data'],
                )
                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            max_shape = [
                [128],
                [128, 128],
                [128, 128, 128],
                [128, 128, 128, 128],
            ]
            min_shape = [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]
            opt_shape = [[32], [32, 32], [32, 32, 32], [32, 32, 32, 32]]

            self.dynamic_shape.min_input_shape = {
                "input_data1": min_shape[len(self.shape1) - 1],
                "input_data2": min_shape[len(self.shape2) - 1],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": max_shape[len(self.shape1) - 1],
                "input_data2": max_shape[len(self.shape2) - 1],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": opt_shape[len(self.shape1) - 1],
                "input_data2": opt_shape[len(self.shape2) - 1],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        if self.shape1[0] == self.shape2[0]:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield self.create_inference_config(), (1, 3), (1e-5, 1e-5)
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield self.create_inference_config(), (1, 3), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (self.create_inference_config(), (1, 3), (1e-05, 1e-05))
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (1, 3), (1e-03, 1e-03))

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertElementwiseTest_one_input_corner_case(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def get_avalible_input_type(self) -> List[np.dtype]:
        return [np.float32, np.float16]

    def sample_program_configs(self):

        # use rand not randn to avoiding pow producing `NAN`
        def generate_weight_corner(is_div=False):
            weight_data = np.random.rand(32).astype(np.float32)
            if is_div:
                weight_data = weight_data + 1.0
            return weight_data

        batch_list = [1, 2, 4]
        shape_list = [[32], [32, 32], [32, 16, 4]]
        op_type_list = [
            'elementwise_add',
            'elementwise_mul',
            'elementwise_sub',
            'elementwise_div',
            'elementwise_pow',
            'elementwise_min',
            'elementwise_max',
        ]
        grid = [batch_list, shape_list, op_type_list]
        for batch, shape, op_type in itertools.product(*grid):
            shape = [batch, *shape]
            axis = -1 if len(shape) == 1 else 1
            is_pow = op_type == 'elementwise_pow'
            is_div = op_type == 'elementwise_div'
            self.op_type = op_type
            self.dims = len(shape)
            dics = [{'axis': axis}]
            ops_config = [
                {
                    'op_type': op_type,
                    'op_inputs': {'X': ['weight'], 'Y': ['input_data']},
                    'op_outputs': {'Out': ['output_data']},
                    'op_attrs': dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={
                    'weight': TensorConfig(
                        data_gen=partial(generate_weight_corner, is_div=is_div)
                    )
                },
                inputs={
                    'input_data': TensorConfig(
                        data_gen=lambda: generate_input(shape, is_pow=is_pow)
                    )
                },
                outputs=['output_data'],
            )
            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            # The input.dims[1] must be equal to the weight's length.
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [4]}
                self.dynamic_shape.max_input_shape = {"input_data": [256]}
                self.dynamic_shape.opt_input_shape = {"input_data": [16]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 32]}
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 32]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32, 4]}
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 32, 256]
                }
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 32, 16]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 32, 4, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 32, 128, 256]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 32, 32, 16]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (self.create_inference_config(), (0, 3), (1e-05, 1e-05))
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (0, 3), (1e-03, 1e-03))
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        if program_config.get_input_type() == np.float32:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            yield (self.create_inference_config(), (1, 2), (1e-05, 1e-05))
        if program_config.get_input_type() == np.float16:
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            yield (self.create_inference_config(), (1, 2), (1e-03, 1e-03))

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()

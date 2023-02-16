/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <array>
#include <memory>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = platform::DataLayout;
using helper = phi::CudnnFrontendConvHelper;

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T>
class FusedDbnApplyOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto cudnn_version = platform::DnnVersion();
    PADDLE_ENFORCE_GE(cudnn_version,
                      8800,
                      phi::errors::PreconditionNotMet(
                          "This op only supports CUDNN version >= 8800, "
                          "but got %d.",
                          cudnn_version));
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));

    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    // Attributes
    bool fuse_dual = ctx.Attr<bool>("fuse_dual");
    // required input variables
    auto* dY_tensor = ctx.Input<Tensor>("dY");
    auto* X_tensor = ctx.Input<Tensor>("X");
    auto* A_tensor = ctx.Input<Tensor>("A");
    auto* B_tensor = ctx.Input<Tensor>("B");
    auto* C_tensor = ctx.Input<Tensor>("C");
    auto* dX_tensor = ctx.Output<Tensor>("dX");
    dX_tensor->mutable_data<T>(ctx.GetPlace());

    // dispensable ios
    const Tensor* X_dual_tensor = nullptr;
    const Tensor* A_dual_tensor = nullptr;
    const Tensor* B_dual_tensor = nullptr;
    const Tensor* C_dual_tensor = nullptr;
    Tensor* dX_dual_tensor = nullptr;
    if (fuse_dual) {
      X_dual_tensor = ctx.Input<Tensor>("X_dual");
      A_dual_tensor = ctx.Input<Tensor>("A_dual");
      B_dual_tensor = ctx.Input<Tensor>("B_dual");
      C_dual_tensor = ctx.Input<Tensor>("C_dual");
      dX_dual_tensor = ctx.Output<Tensor>("dX_dual");
      dX_dual_tensor->mutable_data<T>(ctx.GetPlace());
    }

    // configuration
    using U = BatchNormParamType<T>;
    cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
    auto tensor_format = platform::ToCudnnDataType(
        framework::TransToProtoVarType(dY_tensor->dtype()));
    auto tensor_format_math = CUDNN_DATA_FLOAT;
    auto compute_dtype = CUDNN_DATA_FLOAT;
    // build tensor descriptors
    auto dim_x =
        platform::TransformDimOrder(phi::vectorize<int64_t>(X_tensor->dims()));
    std::vector<int64_t> dim_a(dim_x.size(), 1);
    dim_a[1] = dim_x[1];  //  [1, C, 1, 1]

    std::vector<void*> data_ptrs;
    std::vector<int64_t> uids;
    int64_t uid = 100;

    // inputs
    auto dY_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format, false);
    data_ptrs.push_back(const_cast<T*>(dY_tensor->data<T>()));
    uids.push_back(uid);

    auto X_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format, false);
    data_ptrs.push_back(const_cast<T*>(X_tensor->data<T>()));
    uids.push_back(uid);

    auto A_desc = helper::GetGeneralTensorDescriptor(
        dim_a, layout_format, ++uid, 16, tensor_format_math, false);
    data_ptrs.push_back(const_cast<U*>(A_tensor->data<U>()));
    uids.push_back(uid);

    auto B_desc = helper::GetGeneralTensorDescriptor(
        dim_a, layout_format, ++uid, 16, tensor_format_math, false);
    data_ptrs.push_back(const_cast<U*>(B_tensor->data<U>()));
    uids.push_back(uid);

    auto C_desc = helper::GetGeneralTensorDescriptor(
        dim_a, layout_format, ++uid, 16, tensor_format_math, false);
    data_ptrs.push_back(const_cast<U*>(C_tensor->data<U>()));
    uids.push_back(uid);

    auto X_dual_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format, false);
    if (fuse_dual) {
      data_ptrs.push_back(const_cast<T*>(X_dual_tensor->data<T>()));
      uids.push_back(uid);
    }

    auto A_dual_desc = helper::GetGeneralTensorDescriptor(
        dim_a, layout_format, ++uid, 16, tensor_format_math, false);
    if (fuse_dual) {
      data_ptrs.push_back(const_cast<U*>(A_dual_tensor->data<U>()));
      uids.push_back(uid);
    }

    auto B_dual_desc = helper::GetGeneralTensorDescriptor(
        dim_a, layout_format, ++uid, 16, tensor_format_math, false);
    if (fuse_dual) {
      data_ptrs.push_back(const_cast<U*>(B_dual_tensor->data<U>()));
      uids.push_back(uid);
    }

    auto C_dual_desc = helper::GetGeneralTensorDescriptor(
        dim_a, layout_format, ++uid, 16, tensor_format_math, false);
    if (fuse_dual) {
      data_ptrs.push_back(const_cast<U*>(C_dual_tensor->data<U>()));
      uids.push_back(uid);
    }

    // virtual outputs
    auto after_mul0 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format_math, true);
    auto after_mul1 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format_math, true);
    auto after_add = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format_math, true);

    auto after_mul0_dual = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format_math, true);
    auto after_mul1_dual = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format_math, true);
    auto after_add_dual = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format_math, true);
    // outputs
    auto dX_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format, false);
    data_ptrs.push_back(dX_tensor->data<T>());
    uids.push_back(uid);

    auto dX_dual_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, ++uid, 16, tensor_format, false);
    if (fuse_dual) {
      data_ptrs.push_back(dX_dual_tensor->data<T>());
      uids.push_back(uid);
    }

    // op desc
    auto mul0_op = helper::MakePointwiseOp(
        CUDNN_POINTWISE_MUL, compute_dtype, A_desc, dY_desc, after_mul0);

    auto mul1_op = helper::MakePointwiseOp(
        CUDNN_POINTWISE_MUL, compute_dtype, B_desc, X_desc, after_mul1);

    auto add0_op = helper::MakePointwiseOp(
        CUDNN_POINTWISE_ADD, compute_dtype, after_mul0, after_mul1, after_add);

    auto add1_op = helper::MakePointwiseOp(
        CUDNN_POINTWISE_ADD, compute_dtype, after_add, C_desc, dX_desc);

    auto mul0_op_dual = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                                compute_dtype,
                                                A_dual_desc,
                                                dY_desc,
                                                after_mul0_dual);

    auto mul1_op_dual = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                                compute_dtype,
                                                B_dual_desc,
                                                X_dual_desc,
                                                after_mul1_dual);

    auto add0_op_dual = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                                compute_dtype,
                                                after_mul0_dual,
                                                after_mul1_dual,
                                                after_add_dual);

    auto add1_op_dual = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                                compute_dtype,
                                                after_add_dual,
                                                C_dual_desc,
                                                dX_dual_desc);

    // build op graph
    std::vector<cudnn_frontend::Operation const*> ops = {
        &mul0_op, &mul1_op, &add0_op, &add1_op};
    if (fuse_dual) {
      ops.insert(ops.end(),
                 {&mul0_op_dual, &mul1_op_dual, &add0_op_dual, &add1_op_dual});
    }

    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                        .setHandle(handle)
                        .setOperationGraph(ops.size(), ops.data())
                        .build();
    VLOG(6) << op_graph.describe();

    auto plan = helper::GetPlanByHeuristics(std::move(op_graph), handle);
    auto workspace_size = plan.getWorkspaceSize();
    VLOG(4) << plan.describe() << " requires workspace " << workspace_size;

    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        plan.get_raw_desc(),
                        workspace_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_dbn_apply,
                        ops::FusedDbnApplyOpKernel<paddle::platform::float16>);

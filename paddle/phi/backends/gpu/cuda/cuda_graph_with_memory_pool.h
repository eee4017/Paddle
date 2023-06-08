// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <utility>

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#include "paddle/phi/kernels/funcs/dropout_impl_util.h"
#endif

namespace phi {
namespace backends {
namespace gpu {

#ifdef PADDLE_WITH_CUDA
// The PD_RECORD_CUDA_GRAPH_RANDOM_KERNEL macro is used to manage CUDA kernels 
// that require random seed generation, especially in the context of CUDA graphs. 
// CUDA graphs are sequences of CUDA operations that are captured and can be 
// efficiently replayed. However, each replay of the graph should ideally have a 
// unique sequence of random numbers if randomness is involved in any operation.
// 
// This macro:
// - Checks if the CUDA Graph is currently capturing and if a specified condition is true.
// - Sets up a lambda function for generating and incrementing the random seeds.
// - If the graph is capturing, records the kernel's details for later usage.
// - Executes the CUDA kernel function with the given grid and block sizes, shared memory size,
//   CUDA stream, and kernel arguments.
//
// In summary, this macro helps to ensure that every time the graph is launched, the seed
// for the random number generation in the kernels is correctly set, thus maintaining the 
// intended randomness in the operation.
#define PD_RECORD_CUDA_GRAPH_RANDOM_KERNEL(__cond,                           \
                                           __kernel_func,                    \
                                           __grid,                           \
                                           __block,                          \
                                           __sm_size,                        \
                                           __stream,                         \
                                           __seed_inc,                       \
                                           __seed_expr,                      \
                                           __offset_expr,                    \
                                           ...)                              \
  do {                                                                       \
    VLOG(4) << "PD_RECORD_CUDA_GRAPH_RANDOM_KERNEL" << __FILE__ << " "       \
            << __LINE__;                                                     \
    if (::phi::backends::gpu::CUDAGraph::IsThisThreadCapturing() &&          \
        (__cond)) {                                                          \
      using __Helper =                                                       \
          ::phi::backends::gpu::IsSameKernelHelper<decltype(&__kernel_func), \
                                                   &__kernel_func>;          \
      auto *dev_ctx = ::phi::DeviceContextPool::Instance().GetByPlace(       \
          ::phi::backends::gpu::CUDAGraph::CapturingPlace());                \
      auto __set_seed_func =                                                 \
          [=](::phi::backends::gpu::CUDAKernelParams *__params,              \
              bool __check_only) -> bool {                                   \
        if (__check_only) {                                                  \
          return __params->func() == &__kernel_func &&                       \
                 __Helper::Compare(*__params, __VA_ARGS__);                  \
        }                                                                    \
        auto &KERNEL_PARAMS = *__params;                                     \
        uint64_t __seed, __offset;                                           \
        ::phi::funcs::GetSeedDataAndIncrement(                               \
            *dev_ctx, nullptr, false, 0, __seed_inc, &__seed, &__offset);    \
        __seed_expr = static_cast<decltype(__seed_expr)>(__seed);            \
        __offset_expr = static_cast<decltype(__offset_expr)>(__offset);      \
        return true;                                                         \
      };                                                                     \
      ::phi::backends::gpu::CUDAGraph::RecordRandomKernelInfo(               \
          __set_seed_func);                                                  \
    }                                                                        \
    __kernel_func<<<__grid, __block, __sm_size, __stream>>>(__VA_ARGS__);    \
  } while (0)
#else
#define PD_RECORD_CUDA_GRAPH_RANDOM_KERNEL(__cond,                        \
                                           __kernel_func,                 \
                                           __grid,                        \
                                           __block,                       \
                                           __sm_size,                     \
                                           __stream,                      \
                                           __seed_inc,                    \
                                           __seed_expr,                   \
                                           __offset_expr,                 \
                                           ...)                           \
  do {                                                                    \
    __kernel_func<<<__grid, __block, __sm_size, __stream>>>(__VA_ARGS__); \
  } while (0)
#endif

inline bool IsCUDAGraphCapturing() {
#ifdef PADDLE_WITH_CUDA
  return CUDAGraph::IsCapturing();
#else
  return false;
#endif
}

// Add reset callback if CUDA Graph is capturing.
// Otherwise, invoke callback directly.
template <typename Callback>
inline void AddResetCallbackIfCapturingCUDAGraph(Callback &&callback) {
#ifdef PADDLE_WITH_CUDA
  if (UNLIKELY(IsCUDAGraphCapturing())) {
    return CUDAGraph::AddResetCallbackDuringCapturing(
        std::forward<Callback>(callback));
  }
#endif
  callback();
}

template <typename T>
inline T *RestoreHostMemIfCapturingCUDAGraph(T *host_mem, size_t size) {
  static_assert(std::is_trivial<T>::value, "T must be trivial type");
  static_assert(!std::is_same<T, void>::value, "T cannot be void");
#ifdef PADDLE_WITH_CUDA
  if (UNLIKELY(IsCUDAGraphCapturing())) {
    size_t nbytes = size * sizeof(T);
    void *new_host_mem = new uint8_t[nbytes];
    std::memcpy(new_host_mem, host_mem, nbytes);
    AddResetCallbackIfCapturingCUDAGraph([new_host_mem, size] {
      delete[] reinterpret_cast<uint8_t *>(new_host_mem);
      std::cerr << "Destruct " << new_host_mem << " " << size << std::endl;
    });
    std::cerr << "RestoreHostMemIfCapturingCUDAGraph " << host_mem << " "
              << new_host_mem << " " << size << std::endl;
    return reinterpret_cast<T *>(new_host_mem);
  }
#endif
  return host_mem;
}

}  // namespace gpu
}  // namespace backends
}  // namespace phi

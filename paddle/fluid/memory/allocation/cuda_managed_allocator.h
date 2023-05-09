// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

class CUDAManagedAllocator : public Allocator {
 public:
  std::string DebugAllocatorName() const override { return "CUDAManagedAllocator"; } 
  explicit CUDAManagedAllocator(const platform::CUDAPlace& place)
      : place_(place) {}

  bool IsAllocThreadSafe() const override;

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;

 private:
  platform::CUDAPlace place_;
  std::once_flag once_flag_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

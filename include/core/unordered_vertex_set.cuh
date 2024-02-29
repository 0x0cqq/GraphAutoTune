#pragma once
#include <array>

#include "configs/gpu_consts.cuh"
#include "core/types.hpp"

namespace Core {

// 这个是用来存储已经遍历的节点
template <size_t SIZE>
requires(SIZE < THREADS_PER_WARP)
class UnorderedVertexSet {
    VIndex_t _data[SIZE];

    __device__ void set(VIndex_t val, int pos) { data[pos] = val; }

    __device__ bool has_data(VIndex_t value) {
        const int lid = threadIdx.x % THREADS_PER_WARP;
        bool result = (lid < SIZE) && (_data[lid] == value);
        return __any_sync(0xFFFFFFFF, result);
    }
};

}  // namespace Core
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

  public:
    template <size_t pos>
    __device__ void set(VIndex_t val) {
        _data[pos] = val;
    }

    __device__ VIndex_t get(size_t pos) const { return _data[pos]; }

    __device__ void clear() {
        const int lid = threadIdx.x % THREADS_PER_WARP;
        if (lid < SIZE) _data[lid] = -1;
    }

    __device__ void copy(const UnorderedVertexSet<SIZE>& other) {
        const int lid = threadIdx.x % THREADS_PER_WARP;
        if (lid < SIZE) _data[lid] = other._data[lid];
    }

    template <size_t N>
    __device__ bool has_data(VIndex_t val) const {
        const int lid = threadIdx.x % THREADS_PER_WARP;
        bool result = (lid < N) && (_data[lid] == val);
        return __any_sync(0xFFFFFFFF, result);
    }
};

}  // namespace Core
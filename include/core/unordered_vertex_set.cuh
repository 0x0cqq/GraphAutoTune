#pragma once
#include <array>

#include "configs/launch_config.hpp"
#include "core/types.hpp"

using namespace LaunchConfig;

namespace Core {

// 这个是用来存储已经遍历的节点
template <int SIZE>
requires(SIZE < THREADS_PER_WARP)
class UnorderedVertexSet {
    VIndex_t _data[SIZE];

  public:
    template <int pos>
    __device__ void set(VIndex_t val) {
        _data[pos] = val;
    }

    __device__ VIndex_t get(int pos) const { return _data[pos]; }

    __device__ void clear() {
        const int lid = threadIdx.x % THREADS_PER_WARP;
        if (lid < SIZE) _data[lid] = -1;
    }

    __device__ void copy(const UnorderedVertexSet<SIZE>& other) {
        const int lid = threadIdx.x % THREADS_PER_WARP;
        if (lid < SIZE) _data[lid] = other._data[lid];
    }

    __device__ void copy_single_thread(const UnorderedVertexSet<SIZE>& other) {
#pragma unroll
        for (int i = 0; i < SIZE; i++) {
            _data[i] = other._data[i];
        }
    }

    template <int N>
    __device__ bool has_data(VIndex_t val) const {
        const int lid = threadIdx.x % THREADS_PER_WARP;
        bool result = (lid < N) && (_data[lid] == val);
        return __any_sync(0xFFFFFFFF, result);
    }

    template <int N>
    __device__ bool has_data_single_thread(VIndex_t val) const {
        bool result = false;
#pragma unroll
        for (int i = 0; i < N; i++) {
            result |= (_data[i] == val);
        }
        return result;
    }
};

}  // namespace Core
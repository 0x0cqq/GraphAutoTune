#pragma once
#include <cooperative_groups.h>

#include <array>

#include "core/types.hpp"

namespace cg = cooperative_groups;

namespace Core {

// 这个是用来存储已经遍历的节点
template <int SIZE>
class UnorderedVertexSet {
    VIndex_t _data[SIZE];

  public:
    template <int pos>
    __device__ void set(VIndex_t val) {
        _data[pos] = val;
    }

    __device__ VIndex_t get(int pos) const { return _data[pos]; }

    template <unsigned int WarpSize>
    __device__ void copy(
        const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
        const UnorderedVertexSet<SIZE>& other) {
        const int lid = warp.thread_rank();
#pragma unroll
        for (int i = lid; i < SIZE; i += WarpSize) {
            _data[i] = other._data[i];
        }
    }

    __device__ void copy_single_thread(const UnorderedVertexSet<SIZE>& other) {
#pragma unroll
        for (int i = 0; i < SIZE; i++) {
            _data[i] = other._data[i];
        }
    }

    template <int N, unsigned int WarpSize>
    __device__ bool has_data(
        const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
        VIndex_t val) const {
        const int lid = warp.thread_rank();

        bool result = false;
#pragma unroll
        for (int i = lid; i < N; i += WarpSize) {
            result |= (_data[i] == val);
        }
        return warp.any(result);
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
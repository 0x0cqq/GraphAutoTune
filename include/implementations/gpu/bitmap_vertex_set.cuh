#pragma once
#include <cstdint>
#include <cub/cub.cuh>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "configs/types.hpp"

namespace GPU {

template <Config config>
class BitmapVertexSet {
  private:
    VIndex_t *_data;  // 用 VIndex_t 保持兼容性
    size_t _storage_space;
    VIndex_t _non_zero_cnt;

  public:
    __device__ void __clear() {
        _non_zero_cnt = 0;
        memset(_data, 0, _storage_space * sizeof(VIndex_t));
    }

    __device__ void __init(VIndex_t *input_data, VIndex_t input_size);

    __device__ VIndex_t *__data() const { return _data; }

    __device__ size_t __storage_space() { return _storage_space; }

    __device__ VIndex_t __size() { return _non_zero_cnt; }

    __device__ void __intersect(const BitmapVertexSet &b);
};

}  // namespace GPU

namespace GPU {

__device__ constexpr VIndex_t mask[32] = {
    0x0,        0x1,       0x3,       0x7,       0xF,       0x1F,
    0x3F,       0x7F,      0xFF,      0x1FF,     0x3FF,     0x7FF,
    0xFFF,      0x1FFF,    0x3FFF,    0x7FFF,    0xFFFF,    0x1FFFF,
    0x3FFFF,    0x7FFFF,   0xFFFFF,   0x1FFFFF,  0x3FFFFF,  0x7FFFFF,
    0xFFFFFF,   0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF,
    0x3FFFFFFF, 0x7FFFFFFF};

template <Config config>
__device__ void BitmapVertexSet<config>::__init(VIndex_t *input_data,
                                                VIndex_t input_size) {
    __shared__ uint32_t tmp_val[THREADS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;  // warp id
    int lid = threadIdx.x % THREADS_PER_WARP;  // lane id

    uint32_t *output = tmp_val + wid * THREADS_PER_WARP;

    __clear();
    for (uint32_t i = 0; i < input_size; i++) {
        uint32_t id = input_data[i];
        atomicOr(&_data[id >> 5], 1 << (id & 31));
    }
    _non_zero_cnt = input_size;
};

__device__ uint32_t calculate_non_zero_cnt(VIndex_t *data, size_t size) {
    // warp reduce version
    typedef cub::WarpReduce<uint32_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];
    int wid = threadIdx.x / THREADS_PER_WARP;  // warp id
    int lid = threadIdx.x % THREADS_PER_WARP;  // lane id
    uint32_t sum = 0;
    for (size_t index = 0; index < size; index += THREADS_PER_WARP)
        if (index + lid < size) sum += __popc(data[index + lid]);
    __syncwarp();
    uint32_t aggregate = WarpReduce(temp_storage[wid]).Sum(sum);
    __syncwarp();
    return aggregate;
}

template <Config config>
__device__ void BitmapVertexSet<config>::__intersect(const BitmapVertexSet &b) {
    for (int i = 0; i < _storage_space; i++) {
        _data[i] &= b._data[i];
    }
    _non_zero_cnt = calculate_non_zero_cnt(_data, _storage_space);
    __threadfence_block();
}
};  // namespace GPU
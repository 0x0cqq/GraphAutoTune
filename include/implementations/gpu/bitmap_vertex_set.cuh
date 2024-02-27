#pragma once
#include <cstdint>
#include <cub/cub.cuh>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "core/types.hpp"

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

template <Config config>
constexpr VIndex_t get_storage_space() {
    return (config.graph_config.num_vertices - 1) / (sizeof(VIndex_t) * 8) + 1;
}

template <Config config>
void prepare_bitmap_data_cpu(const VIndex_t *data, VIndex_t size,
                             VIndex_t *output) {
    static_assert(config.vertex_set_config.vertex_store_type == Bitmap);
    constexpr VIndex_t storage_size = get_storage_space<config>();

    memset(output, 0, storage_size * sizeof(VIndex_t));

    constexpr int B = sizeof(VIndex_t) + 1;
    constexpr int M = (1 << B) - 1;

    for (VIndex_t i = 0; i < size; i++) {
        output[data[i] >> B] |= 1 << (data[i] & M);
    }
}

__device__ uint32_t calculate_non_zero_cnt(VIndex_t *data, size_t size) {
    // warp reduce version
    typedef cub::WarpReduce<uint32_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];
    const int wid = threadIdx.x / THREADS_PER_WARP;  // warp id
    const int lid = threadIdx.x % THREADS_PER_WARP;  // lane id
    uint32_t sum = 0;
    for (size_t index = 0; index < size; index += THREADS_PER_WARP)
        if (index + lid < size) sum += __popc(data[index + lid]);
    __syncwarp();
    uint32_t aggregate = WarpReduce(temp_storage[wid]).Sum(sum);
    __syncwarp();
    return aggregate;
}

template <Config config>
__device__ void BitmapVertexSet<config>::__init(VIndex_t *input_data,
                                                VIndex_t input_size) {
    static_assert(config.vertex_set_config.vertex_store_type == Bitmap);
    // 输入的就是一个 vertex set。
    const int lid = threadIdx.x % THREADS_PER_WARP;  // lane id
    int t = calculate_non_zero_cnt(input_data, input_size);
    if (lid == 0) {
        _non_zero_cnt = t;
        _data = input_data;
        _storage_space = input_size;
    }
    __threadfence_block();
};

template <Config config>
__device__ void BitmapVertexSet<config>::__intersect(const BitmapVertexSet &b) {
    const int wid = threadIdx.x / THREADS_PER_WARP;  // warp id
    const int lid = threadIdx.x % THREADS_PER_WARP;  // lane id
    for (int index = 0; index < _storage_space; index += THREADS_PER_WARP) {
        if (index + lid < _storage_space)
            _data[index + lid] &= b._data[index + lid];
    }
    __threadfence_block();
    int t = calculate_non_zero_cnt(_data, _storage_space);
    if (lid == 0) _non_zero_cnt = t;
    __threadfence_block();
}
};  // namespace GPU
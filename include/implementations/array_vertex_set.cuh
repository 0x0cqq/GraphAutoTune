#pragma once

#include <array>
#include <concepts>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "core/types.hpp"

namespace GPU {

// 用数组形式存储的节点集合。
template <Config config>
class ArrayVertexSet {
  private:
    VIndex_t _size;
    VIndex_t _allocated_size;
    VIndex_t* _data;

  public:
    __device__ void __init(VIndex_t* input_data, VIndex_t input_size);

    __device__ VIndex_t __size() const { return _size; }

    __device__ VIndex_t* __data() const { return _data; }

    __device__ void __clear() { _size = 0; }

    __device__ size_t __storage_space() const { return _allocated_size; }

    __device__ void __intersect(const ArrayVertexSet<config>& b);
};

}  // namespace GPU

namespace GPU {

inline __device__ bool binary_search(const VIndex_t u, const VIndex_t* b,
                                     const VIndex_t nb) {
    int mid, l = 0, r = int(nb) - 1;
    while (l <= r) {
        mid = (l + r) >> 1;
        if (b[mid] < u) {
            l = mid + 1;
        } else if (b[mid] > u) {
            r = mid - 1;
        } else {
            return true;
        }
    }
    return false;
}

inline __device__ bool linear_search(const VIndex_t u, const VIndex_t* b,
                                     const VIndex_t nb) {
    for (int idx_b = 0; idx_b < nb; idx_b++) {
        if (b[idx_b] == u) {
            return true;
        } else if (b[idx_b] > u) {
            return false;
        }
    }
    return false;
}

template <Config config>
inline __device__ bool search_dispatcher(VIndex_t u, const VIndex_t* b,
                                         VIndex_t nb) {
    if constexpr (config.vertex_set_config.set_search_type == Binary) {
        return binary_search(u, b, nb);
    } else {
        return linear_search(u, b, nb);
    }
}

template <Config config>
__device__ VIndex_t do_intersection_parallel(VIndex_t* out, const VIndex_t* a,
                                             const VIndex_t* b, VIndex_t na,
                                             VIndex_t nb) {
    __shared__ VIndex_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ VIndex_t block_out_size[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;  // warp id
    int lid = threadIdx.x % THREADS_PER_WARP;  // lane id
    VIndex_t* out_offset = block_out_offset + wid * THREADS_PER_WARP;
    VIndex_t& out_size = block_out_size[wid];

    if (lid == 0) out_size = 0;

    for (int num_done = 0; num_done < na; num_done += THREADS_PER_WARP) {
        bool found = false;
        VIndex_t u = 0;
        if (num_done + lid < na) {
            u = a[num_done + lid];  // u: an element in set a
            found = search_dispatcher<config>(u, b, nb);
        }
        out_offset[lid] = found;
        __threadfence_block();

#pragma unroll
        for (int s = 1; s < THREADS_PER_WARP; s *= 2) {
            uint32_t v = lid >= s ? out_offset[lid - s] : 0;
            __threadfence_block();
            out_offset[lid] += v;
            __threadfence_block();
        }

        if (found) {
            uint32_t offset = out_offset[lid] - 1;
            out[out_size + offset] = u;
        }

        if (lid == 0) out_size += out_offset[THREADS_PER_WARP - 1];
    }

    __threadfence_block();
    return out_size;
}

template <Config config>
__device__ VIndex_t do_intersection_serial(VIndex_t* out, const VIndex_t* a,
                                           const VIndex_t* b, VIndex_t na,
                                           VIndex_t nb) {
    int wid = threadIdx.x / THREADS_PER_WARP;  // warp id
    int lid = threadIdx.x % THREADS_PER_WARP;  // lane id
    int out_size = 0;

    for (int num_done = 0; num_done < na; num_done++) {
        bool found = 0;
        VIndex_t u = 0;
        if (num_done + lid < na) {
            u = a[num_done];  // u: an element in set a
            search_dispatcher<config>(u, b, nb);
        }
        if (found) out[out_size++] = u;
    }
    return out_size;
}

template <Config config>
__device__ void ArrayVertexSet<config>::__init(VIndex_t* input_data,
                                               VIndex_t input_size) {
    static_assert(config.vertex_set_config.vertex_store_type == Array);

    const int lid = threadIdx.x % THREADS_PER_WARP;
    if (lid == 0) {
        _data = input_data, _allocated_size = _size = input_size;
    }
}

template <Config config>
__device__ VIndex_t do_intersection_dispatcher(VIndex_t* out, const VIndex_t* a,
                                               const VIndex_t* b, VIndex_t na,
                                               VIndex_t nb) {
    if constexpr (config.vertex_set_config.set_intersection_type == Parallel) {
        return do_intersection_parallel<config>(out, a, b, na, nb);
    } else {
        return do_intersection_serial<config>(out, a, b, na, nb);
    }
}

template <Config config>
__device__ void ArrayVertexSet<config>::__intersect(
    const ArrayVertexSet<config>& b) {
    _size = do_intersection_dispatcher<config>(__data(), __data(), b.__data(),
                                               __size(), b.__size());
}

}  // namespace GPU
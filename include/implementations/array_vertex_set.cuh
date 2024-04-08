#pragma once

#include <cooperative_groups.h>

#include <array>
#include <concepts>
#include <nvfunctional>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "core/types.hpp"
#include "core/unordered_vertex_set.cuh"

namespace GPU {

__device__ unsigned long long counter = 0;

// 用数组形式存储的节点集合。
template <Config config>
class ArrayVertexSet {
  private:
    VIndex_t _size;
    VIndex_t _allocated_size;
    VIndex_t* _data;

  public:
    __device__ void init_empty(VIndex_t* space, VIndex_t storage_size) {
        static_assert(config.vertex_set_config.vertex_store_type == Array);
        const int lid = threadIdx.x % THREADS_PER_WARP;
        if (lid == 0) {
            _data = space, _allocated_size = storage_size, _size = 0;
        }
    }

    __device__ void init(VIndex_t* input_data, VIndex_t input_size) {
        static_assert(config.vertex_set_config.vertex_store_type == Array);
        const int lid = threadIdx.x % THREADS_PER_WARP;
        if (lid == 0) {
            _data = input_data, _allocated_size = _size = input_size;
        }
    }

    __device__ void init_copy(const VIndex_t* input_data, VIndex_t input_size) {
        static_assert(config.vertex_set_config.vertex_store_type == Array);
        const int lid = threadIdx.x % THREADS_PER_WARP;
        assert(_allocated_size >= input_size);
        for (VIndex_t base = 0; base < input_size; base += THREADS_PER_WARP) {
            VIndex_t i = base + lid;
            if (i < input_size) _data[i] = input_data[i];
        }
        if (lid == 0) {
            _size = input_size;
        }
    }

    __device__ inline VIndex_t get(VIndex_t idx) const { return _data[idx]; }

    __device__ inline VIndex_t size() const { return _size; }

    __device__ inline VIndex_t* data() const { return _data; }

    __device__ inline void clear() { _size = 0; }

    __device__ inline size_t storage_space() const { return _allocated_size; }

    template <int depth, size_t SIZE>
    __device__ VIndex_t
    subtraction_size_single_thread(const Core::UnorderedVertexSet<SIZE>& set);

    // 对 Output 没有任何假设，除了空间是够的之外。
    // TODO: 或可以用 __restrict__ 修饰符加速
    __device__ void intersect(const ArrayVertexSet& a, const ArrayVertexSet& b);
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
            u = a[num_done + lid];
            search_dispatcher<config>(u, b, nb);
        }
        if (found) out[out_size++] = u;
    }
    return out_size;
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
__device__ void ArrayVertexSet<config>::intersect(const ArrayVertexSet& a,
                                                  const ArrayVertexSet& b) {
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    if (lane_id == 0) {
        atomicAdd(&counter, 1);
    }
    __syncthreads();

    VIndex_t after_intersect_size = do_intersection_dispatcher<config>(
        this->data(), a.data(), b.data(), a.size(), b.size());
    this->_size = after_intersect_size;
    assert(_allocated_size >= _size);
}

template <Config config>
template <int depth, size_t SIZE>
__device__ VIndex_t ArrayVertexSet<config>::subtraction_size_single_thread(
    const Core::UnorderedVertexSet<SIZE>& set) {
    // 这是只有一个线程的 version
    int ans = 0;
    for (int i = 0; i < depth; i++) {
        if (search_dispatcher<config>(set.get(i), _data, _size)) {
            ans++;
        }
    }
    return _size - ans;
}

}  // namespace GPU

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#include <array>
#include <concepts>
#include <cub/cub.cuh>
#include <nvfunctional>

#include "configs/config.hpp"
#include "consts/general_consts.hpp"
#include "core/types.hpp"
#include "core/unordered_vertex_set.cuh"
#include "utils/cuda_utils.cuh"

namespace cg = cooperative_groups;

namespace GPU {

__device__ unsigned long long counter = 0;

// 用数组形式存储的节点集合。
template <Config config>
class ArrayVertexSet {
  private:
    // 前两者是分配的空间
    VIndex_t _allocated_size;
    VIndex_t* _space;
    // 后两者是实际使用的空间
    VIndex_t _size;
    VIndex_t* _data;

  public:
    template <unsigned int WarpSize>
    __device__ void init_empty(
        const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
        VIndex_t* space, VIndex_t storage_size) {
        static_assert(config.vertex_set_config.vertex_store_type == Array);
        if (warp.thread_rank() == 0) {
            _space = space, _data = space;
            _allocated_size = storage_size, _size = 0;
        }
    }

    __device__ void init_empty(VIndex_t* space, VIndex_t storage_size) {
        static_assert(config.vertex_set_config.vertex_store_type == Array);
        _space = space, _data = space;
        _allocated_size = storage_size, _size = 0;
    }

    // 是用来临时创建空间的，比如给 neighbor vertex 临时给一个 Vertex Set
    template <unsigned int WarpSize>
    __device__ void init(
        const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
        VIndex_t* input_data, VIndex_t input_size) {
        static_assert(config.vertex_set_config.vertex_store_type == Array);
        if (warp.thread_rank() == 0) {
            _data = input_data, _size = input_size;
            _space = nullptr, _allocated_size = 0;
        }
    }

    template <unsigned int WarpSize>
    __device__ void use_copy(
        cg::thread_block_tile<WarpSize, cg::thread_block> warp,
        VIndex_t* input_data, VIndex_t input_size) {
        static_assert(config.vertex_set_config.vertex_store_type == Array);
        if (warp.thread_rank() == 0) {
            _size = input_size, _data = input_data;
        }
    }

    __device__ void use_copy(VIndex_t* input_data, VIndex_t input_size) {
        static_assert(config.vertex_set_config.vertex_store_type == Array);
        _size = input_size, _data = input_data;
    }

    __device__ inline VIndex_t get(VIndex_t idx) const { return _data[idx]; }

    __device__ inline VIndex_t size() const { return _size; }

    __device__ inline VIndex_t* data() const { return _data; }

    __device__ inline void clear() { _size = 0; }

    __device__ inline int storage_space() const { return _allocated_size; }

    template <int depth, int SIZE>
    __device__ VIndex_t
    subtraction_size_onethread(const Core::UnorderedVertexSet<SIZE>& set);

    // 对 Output 没有任何假设，除了空间是够的之外。
    template <unsigned int WarpSize>
    __device__ void intersect(
        const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
        ArrayVertexSet& a, const ArrayVertexSet& b);

    // size 会保存到当前的 vertex
    template <int depth, int SIZE, unsigned int WarpSize>
    __device__ void intersect_size(
        const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
        const ArrayVertexSet& a, const ArrayVertexSet& b,
        const Core::UnorderedVertexSet<SIZE>& set);
};

}  // namespace GPU

namespace GPU {

// 用二进制的方法重写这个二分，这样就没准可以循环展开了
__device__ bool binary_search(const VIndex_t u, const VIndex_t* b,
                              const VIndex_t nb) {
    if (nb == 0) return false;
    // 获取 nb 最高位的二进制位数
    const VIndex_t p = 32 - __clz(nb - 1);
    VIndex_t n = 0;
#pragma unroll
    // 每次决定一个二进制位，从高到低
    for (int i = p - 1; i >= 0; i--) {
        // 这次决定的是从高往低的第 i 位
        const VIndex_t index = n | (1 << i);
        // 往右侧走
        VIndex_t x = b[index];
        if (index < nb && x <= u) {
            n = index;
        }
    }
    return b[n] == u;
}

__device__ bool linear_search(const VIndex_t u, const VIndex_t* b,
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
__device__ bool search_dispatcher(VIndex_t u, const VIndex_t* b, VIndex_t nb) {
    if constexpr (config.vertex_set_config.set_search_type == 0) {
        return binary_search(u, b, nb);
    } else if constexpr (config.vertex_set_config.set_search_type == INT_MAX) {
        return linear_search(u, b, nb);
    } else {
        if (nb < config.vertex_set_config.set_search_type) {
            return linear_search(u, b, nb);
        } else {
            return binary_search(u, b, nb);
        }
    }
}

template <Config config, unsigned int WarpSize>
__device__ VIndex_t do_intersection_parallel(
    const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
    VIndex_t* __restrict__ out, const VIndex_t* __restrict__ a,
    const VIndex_t* __restrict__ b, VIndex_t na, VIndex_t nb) {
    constexpr auto launch_config = config.engine_config.launch_config;
    constexpr int warps_per_block =
        launch_config.threads_per_block / launch_config.threads_per_warp;

    __shared__ VIndex_t block_out_size[warps_per_block];

    if (na > nb) {
        swap(na, nb);
        swap(a, b);
    }

    const int wid = warp.meta_group_rank();  // warp id
    const int lid = warp.thread_rank();      // lane id
    VIndex_t& out_size = block_out_size[wid];

    if (lid == 0) out_size = 0;

    for (int base = 0; base < na; base += WarpSize) {
        int found = 0;
        VIndex_t u = 0;
        if (base + lid < na) {
            u = a[base + lid];  // u: an element in set a
            found = int(search_dispatcher<config>(u, b, nb));
        }

        const int offset = cg::inclusive_scan(warp, found);

        warp.sync();
        __threadfence_block();
        if (found) out[out_size + offset - 1] = u;
        if (lid == WarpSize - 1) out_size += offset;
    }

    __threadfence_block();
    return out_size;
}

template <Config config, unsigned int WarpSize>
__device__ VIndex_t do_intersection_serial(
    const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
    VIndex_t* __restrict__ out, const VIndex_t* __restrict__ a,
    const VIndex_t* __restrict__ b, VIndex_t na, VIndex_t nb) {
    int out_size = 0;

    for (int num_done = 0; num_done < na; num_done++) {
        bool found = false;
        VIndex_t u = a[num_done];
        found = search_dispatcher<config>(u, b, nb);
        if (found) out[out_size++] = u;
    }
    return out_size;
}

template <Config config, unsigned int WarpSize>
__device__ VIndex_t do_intersection_dispatcher(
    const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
    VIndex_t* __restrict__ out, const VIndex_t* __restrict__ a,
    const VIndex_t* __restrict__ b, VIndex_t na, VIndex_t nb) {
    if constexpr (config.vertex_set_config.set_intersection_type == Parallel) {
        return do_intersection_parallel<config>(warp, out, a, b, na, nb);
    } else {
        return do_intersection_serial<config>(warp, out, a, b, na, nb);
    }
}

template <Config config, int depth, int SIZE, unsigned int WarpSize>
__device__ VIndex_t do_intersection_serial_size(
    const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
    const VIndex_t* a, const VIndex_t* b, VIndex_t na, VIndex_t nb,
    const Core::UnorderedVertexSet<SIZE>& set) {
    int out_size = 0;

    VIndex_t u = 0;
    bool found = false;
    for (int num_done = 0; num_done < na; num_done++) {
        u = a[num_done];
        found =
            search_dispatcher<config>(u, b, nb) & !set.has_data<depth>(warp, u);
        if (found) out_size++;
    }
    return out_size;
}

template <Config config, int depth, int SIZE, unsigned int WarpSize>
inline __device__ VIndex_t do_intersection_parallel_size(
    const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
    const VIndex_t* a, const VIndex_t* b, VIndex_t na, VIndex_t nb,
    const Core::UnorderedVertexSet<SIZE>& set) {
    // na 应该是元素个数比较小的那个
    if (na > nb) {
        swap(na, nb);
        swap(a, b);
    }

    int lid = warp.thread_rank();

    VIndex_t out_size = 0;
    for (int base = 0; base < na; base += WarpSize) {
        bool found = false;
        if (base + lid < na) {
            VIndex_t u = a[base + lid];  // u: an element in set a
            found = !set.has_data_single_thread<depth>(u) &&
                    search_dispatcher<config>(u, b, nb);
        }
        out_size += found;
    }
    warp.sync();
    VIndex_t aggregate = cg::reduce(warp, out_size, cg::plus<VIndex_t>());
    // only available in lid == 0
    return aggregate;
}

template <Config config, int depth, int SIZE, unsigned int WarpSize>
__device__ VIndex_t do_intersection_dispatcher_size(
    const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
    const VIndex_t* a, const VIndex_t* b, VIndex_t na, VIndex_t nb,
    const Core::UnorderedVertexSet<SIZE>& set) {
    if constexpr (config.vertex_set_config.set_intersection_type == Parallel) {
        return do_intersection_parallel_size<config, depth>(warp, a, b, na, nb,
                                                            set);
    } else {
        return do_intersection_serial_size<config, depth>(warp, a, b, na, nb,
                                                          set);
    }
}
template <Config config>
template <unsigned int WarpSize>
__device__ void ArrayVertexSet<config>::intersect(
    const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
    ArrayVertexSet<config>& a, const ArrayVertexSet<config>& b) {
    VIndex_t after_intersect_size = do_intersection_dispatcher<config>(
        warp, this->_space, a.data(), b.data(), a.size(), b.size());

    if (warp.thread_rank() == 0) {
        this->_data = this->_space;
        this->_size = after_intersect_size;
    }
}

template <Config config>
template <int depth, int SIZE, unsigned int WarpSize>
__device__ void ArrayVertexSet<config>::intersect_size(
    const cg::thread_block_tile<WarpSize, cg::thread_block>& warp,
    const ArrayVertexSet<config>& a, const ArrayVertexSet<config>& b,
    const Core::UnorderedVertexSet<SIZE>& set) {
    // 一个低成本的计算交集大小的function
    VIndex_t size = do_intersection_dispatcher_size<config, depth>(
        warp, a.data(), b.data(), a.size(), b.size(), set);
    if (warp.thread_rank() == 0) {
        this->_size = size;
    }
}

template <Config config>
template <int depth, int SIZE>
__device__ VIndex_t ArrayVertexSet<config>::subtraction_size_onethread(
    const Core::UnorderedVertexSet<SIZE>& set) {
    // 这是只有一个线程的 version
    int ans = 0;
#pragma unroll
    for (int i = 0; i < depth; i++) {
        if (search_dispatcher<config>(set.get(i), _data, _size)) {
            ans++;
        }
    }
    return _size - ans;
}

}  // namespace GPU

#pragma once
#include <array>
#include <chrono>
#include <cub/cub.cuh>
#include <iostream>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "core/graph.cuh"
#include "core/schedule.hpp"
#include "core/types.hpp"
#include "core/vertex_set.cuh"
#include "engine/context.cuh"
#include "engine/engine_kernel.cuh"
#include "engine/storage.cuh"
#include "engine/worker.cuh"

namespace Engine {

template <Config config>
class Executor {
  private:
    using VertexSet = VertexSetTypeDispatcher<config>::type;

    // 搜索过程中每层所需要的临时存储。
    DeviceType _device_type;
    PrefixStorages<config> prefix_storages;
    VertexStorages<config> vertex_storages;

    DeviceContext<config> *device_context;
    int cur_progress[MAX_VERTEXES];
    unsigned long long ans;

    // CPU 端的搜索
    template <int cur_pattern_vid>
    __host__ void search();

    template <int cur_pattern_vid>
    __host__ void prepare();

    template <int cur_pattern_vid>
    __host__ bool extend();

    __host__ void final_step();

  public:
    __host__ Executor(DeviceType device_type)
        : _device_type(device_type), device_context(nullptr) {
#ifndef NDEBUG
        std::cerr << "Executor Constructor, ";
        std::cerr << "Device Type: " << device_type << std::endl;
#endif

        // 给 prefix storage 里面的 vertex 分配一个标准的内存空间
        for (int i = 0; i < MAX_PREFIXS; i++) {
            prefix_storages[i].init();
            set_vertex_set_space<config>
                <<<num_blocks, THREADS_PER_BLOCK>>>(prefix_storages[i]);
        }
        for (int i = 0; i < MAX_VERTEXES; i++) {
            vertex_storages[i].init();
        }

        gpuErrchk(cudaDeviceSynchronize());

#ifndef NDEBUG
        std::cerr << "Executor Constructor Done" << std::endl;
#endif
    }

    __host__ ~Executor() {
        for (int i = 0; i < MAX_PREFIXS; i++) {
            prefix_storages[i].destroy();
        }
        for (int i = 0; i < MAX_VERTEXES; i++) {
            vertex_storages[i].destroy();
        }
    }

    __host__ void set_context(DeviceContext<config> &context) {
        device_context = &context;
    }

    __host__ void do_extend_size_sum(VertexStorage<config> &v_storage) {
        int num_units = v_storage.num_units;
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        gpuErrchk(cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, v_storage.unit_extend_size,
            v_storage.unit_extend_sum, num_units));

        // Allocate temporary storage for inclusive prefix sum
        gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Run inclusive prefix sum
        gpuErrchk(cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, v_storage.unit_extend_size,
            v_storage.unit_extend_sum, num_units));

        gpuErrchk(cudaFree(d_temp_storage));
    }

    __host__ unsigned long long perform_search(DeviceContext<config> &context);
};

// 这个函数就约等于从 V 全集的只有一个 Unit 开始的 Extend
template <Config config>
__host__ unsigned long long Executor<config>::perform_search(
    DeviceContext<config> &context) {
    // 把 Context 到当前的 Executor
    this->set_context(context);
    // 重置答案
    this->ans = 0;
    // 在这里做第一层，也就是选第一个点，更新
    VIndex_t v_cnt = device_context->graph_backend.v_cnt();
    for (VIndex_t base_index = 0; base_index < v_cnt; base_index += NUMS_UNIT) {
        VIndex_t start = base_index,
                 end = std::min(v_cnt, base_index + NUMS_UNIT);
#ifndef NDEBUG
        std::cerr << "First Extend Kernel, start: " << start << ", end: " << end
                  << std::endl;
#endif
        first_extend_kernel<config><<<num_blocks, THREADS_PER_BLOCK>>>(
            *device_context, prefix_storages[0], vertex_storages[0], start,
            end);

        vertex_storages[0].num_units = end - start;

        search<0>();
    }
    return this->ans;
}

template <Config config>
template <int cur_pattern_vid>
__host__ void Executor<config>::search() {
#ifndef NDEBUG
    auto time_start = std::chrono::high_resolution_clock::now();
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cerr << "Search, enter level " << cur_pattern_vid << std::endl;
    }
#endif

    // 如果已经处理完了不需要 IEP 的节点，进入最终的 IEP 处理
    if (cur_pattern_vid == device_context->schedule_data.basic_vertexes) {
        final_step();
        return;
    }

    prepare<cur_pattern_vid>();

    cur_progress[cur_pattern_vid] = 0;

    // 直到上一层全部被拓展完完成
    while (extend<cur_pattern_vid + 1>()) {
        // 对下一层递归进行搜素
        // 这句 if constexpr 不能缺少，否则会导致编译器无限递归
        // 不用特化的原因是，模板类中，成员函数无法单独特化
        if constexpr (cur_pattern_vid + 1 < MAX_VERTEXES) {
            search<cur_pattern_vid + 1>();
        }
    }
#ifndef NDEBUG
    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start);
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cerr << "Search, leave level " << cur_pattern_vid << ", time "
                  << duration.count() << " ms" << std::endl;
    }
#endif
}

// 计算 v_storage 的 cur_pattern_vid 层的 prefix sum
template <Config config>
template <int cur_pattern_vid>
__host__ void Executor<config>::prepare() {
#ifndef NDEBUG
    std::cerr << "Enter prepare at level " << cur_pattern_vid << std::endl;
#endif
    // 计算扩展相关的 size
    extend_v_storage<config, cur_pattern_vid>
        <<<num_blocks, THREADS_PER_BLOCK>>>(*device_context, prefix_storages,
                                            vertex_storages);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    // 做前缀和
    do_extend_size_sum(vertex_storages[cur_pattern_vid]);

#ifndef NDEBUG
    std::cerr << "Leave prepare at level " << cur_pattern_vid << std::endl;
#endif
}

// 扩展到 cur_pattern_vid 那一层的 Vertex Storage 和对应的些许的 Prefix vid
template <Config config>
template <int cur_pattern_vid>
__host__ bool Executor<config>::extend() {
#ifndef NDEBUG
    auto time_start = std::chrono::high_resolution_clock::now();
    if (cur_pattern_vid < LOG_DEPTH) {
        // std::cerr << "Enter Host Extend at level " <<
        // cur_pattern_vertex_id
        //           << ", progress at " << last.cur_unit() << "/"
        //           << last.alloc_units() << std::endl;
    }
#endif

    // 在 某个地方（？）记录一下已经处理了几个节点了
    // 检查是否需要拓展

    int cur_unit = cur_progress[cur_pattern_vid - 1];
    // 如果不需要拓展就直接返回 false
    int *next_unit;
    gpuErrchk(cudaMallocManaged(&next_unit, sizeof(int)));
    get_next_unit<config><<<1, 1>>>(cur_unit, next_unit, NUMS_UNIT,
                                    vertex_storages[cur_pattern_vid - 1]);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    if (*next_unit == cur_unit) return false;
#ifndef NDEBUG
    std::cerr << "Extend: [" << cur_unit << "," << *next_unit << ")"
              << std::endl;
#endif
    extend_p_storage<config, cur_pattern_vid>
        <<<num_blocks, THREADS_PER_BLOCK>>>(*device_context, prefix_storages,
                                            vertex_storages, cur_unit,
                                            *next_unit);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cur_progress[cur_pattern_vid] = *next_unit;

    gpuErrchk(cudaFree(next_unit));
    // 清除下一层的存储
    // current.clear();
    // 重新分配存储
    // current.reallocate();

#ifndef NDEBUG
    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start);
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cerr << "Leave Host Extend at level " << cur_pattern_vid
                  << ", time " << duration.count() << " ms" << std::endl;
    }
#endif

    return true;
}

template <Config config>
__host__ void Executor<config>::final_step() {
    // 先不加 IEP
    int last_prefix_id =
        this->device_context->schedule_data.total_prefix_num - 1;

#ifndef NDEBUG
    auto time_start = std::chrono::high_resolution_clock::now();
#endif
    ans += 1;  // Tmp
#ifndef NDEBUG
    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start);
#endif
}

}  // namespace Engine

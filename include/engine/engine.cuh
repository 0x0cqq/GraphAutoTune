#pragma once
#include <nvtx3/nvToolsExt.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>

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
    DeviceType _device_type;

    // 搜索过程中每层所需要的临时存储。
    PrefixStorages<config> prefix_storages;
    VertexStorages<config> vertex_storages;

    // 用于 IEP 的临时存储
    unsigned long long *d_ans;

    // 用于 cub device prefix 的空间
    void *d_prefix_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 存储一些恒定的 Context，例如 Schedule 和 Graph Backend
    DeviceContext<config> *device_context;

    template <int cur_pattern_vid>
    __host__ void search();

    template <int cur_pattern_vid>
    __host__ void prepare(VIndex_t &extend_total_units);

    template <int cur_pattern_vid>
    __host__ void extend(int base_extend_unit_id, int num_extend_units);

    template <int cur_pattern_vid>
    __host__ void final_step();

  public:
    __host__ Executor(DeviceType device_type)
        : _device_type(device_type), device_context(nullptr) {
#ifndef NDEBUG
        std::cout << "Executor: Constructor, device type: "
                  << (device_type == GPU_DEVICE ? "GPU" : "CPU") << std::endl;
#endif

        for (int i = 0; i < MAX_PREFIXS; i++) {
            prefix_storages[i].init();
            // 给 prefix storage 里面的 vertex set 分配内存空间
            set_vertex_set_space_kernel<config>
                <<<num_blocks, THREADS_PER_BLOCK>>>(prefix_storages[i]);
        }
        for (int i = 0; i < MAX_VERTEXES; i++) {
            vertex_storages[i].init();
        }

        // 分配答案空间
        gpuErrchk(cudaMalloc(&d_ans, sizeof(unsigned long long) * NUMS_UNIT));
        // 设备空间设置为 0
        gpuErrchk(cudaMemset(d_ans, 0, sizeof(unsigned long long) * NUMS_UNIT));

        // 准备前缀和空间
        VIndex_t *temp = nullptr;
        gpuErrchk(cub::DeviceScan::InclusiveSum(
            d_prefix_temp_storage, temp_storage_bytes, temp, temp, NUMS_UNIT));
        gpuErrchk(cudaMalloc(&d_prefix_temp_storage, temp_storage_bytes));

        gpuErrchk(cudaDeviceSynchronize());

#ifndef NDEBUG
        std::cout << "Executor: Constructor Done. ";
        print_device_memory();
#endif
    }

    __host__ ~Executor() {
        for (int i = 0; i < MAX_PREFIXS; i++) {
            prefix_storages[i].destroy();
        }
        for (int i = 0; i < MAX_VERTEXES; i++) {
            vertex_storages[i].destroy();
        }

        gpuErrchk(cudaFree(d_ans));
        gpuErrchk(cudaFree(d_prefix_temp_storage));
    }

    __host__ void set_context(DeviceContext<config> &context) {
        device_context = &context;
    }

    // 通过 thrust::reduce 求和所有的 Unit 的 d_ans
    __host__ unsigned long long reduce_answer() {
        thrust::device_ptr<unsigned long long> ans_ptr(d_ans);
        return thrust::reduce(ans_ptr, ans_ptr + NUMS_UNIT);
    }

    __host__ unsigned long long perform_search(DeviceContext<config> &context);
};

// 这个函数约等于从 V 全集的只有一个 Unit 开始的 Extend，并且开启后面的搜索
template <Config config>
__host__ unsigned long long Executor<config>::perform_search(
    DeviceContext<config> &context) {
    // 把 Context 到当前的 Executor
    this->set_context(context);

    // 设备空间设置为0
    // 按理说这个应该放到外面。但总共也就一次，所以就放在这里了
    gpuErrchk(cudaMemset(d_ans, 0, sizeof(unsigned long long) * NUMS_UNIT));

    VIndex_t v_cnt = device_context->graph_backend.v_cnt();
    for (VIndex_t base_index = 0; base_index < v_cnt; base_index += NUMS_UNIT) {
        VIndex_t start_vid = base_index,
                 end_vid = std::min(v_cnt, base_index + NUMS_UNIT);
#ifndef NDEBUG
        std::cout << "First Extend Kernel, start: " << start_vid
                  << ", end: " << end_vid << std::endl;
#endif
        // 构建 prefix_storages[0] 和 vertex_storages[0]，也就是 prefix = [0]
        first_extend_kernel<config><<<num_blocks, THREADS_PER_BLOCK>>>(
            *device_context, prefix_storages[0], vertex_storages[0], start_vid,
            end_vid);

        // vertex_storages[0].num_units
        vertex_storages[0].num_units = end_vid - start_vid;

        search<0>();
    }

    // 从 d_ans 里面把答案取出来
    return reduce_answer();
}

template <Config config>
template <int cur_pattern_vid>
__host__ void Executor<config>::search() {
#ifndef NDEBUG
    auto time_start = std::chrono::high_resolution_clock::now();
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cout << "Level " << cur_pattern_vid << ": enter search"
                  << std::endl;
    }
#endif

    // 如果已经处理完了前 basic_vertexes 个点，那么就进行最后的处理
    if (cur_pattern_vid == device_context->schedule_data.basic_vertexes - 1) {
        final_step<cur_pattern_vid>();
        return;
    }

    // 构建 v_storages[cur_pattern_vid] 的 unit_extend_size & sum
    // 也就是向 cur_pattern_vid + 1 扩展的前置工作
    VIndex_t extend_total_units = 0;
    prepare<cur_pattern_vid>(extend_total_units);

    // 等待下一回合的结果被拷贝回来
    gpuErrchk(cudaStreamSynchronize(0));

    // std::cout << "Extend Total Units: " << extend_total_units << std::endl;

    // 直到本层全部被拓展完完成
    for (int base = 0; base < extend_total_units; base += NUMS_UNIT) {
        int num = min(extend_total_units - base, NUMS_UNIT);
        extend<cur_pattern_vid>(base, num);
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
        std::cout << "Level " << cur_pattern_vid << ": leave search, time "
                  << duration.count() << " ms" << std::endl;
    }
#endif
}

// 这个函数构建 cur_pattern_vid 位置的 unit_extend_size & sum
template <Config config>
template <int cur_pattern_vid>
__host__ void Executor<config>::prepare(VIndex_t &extend_total_units) {
#ifndef NDEBUG
    auto start = std::chrono::high_resolution_clock::now();
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cout << "Level " << cur_pattern_vid << ": enter prepare."
                  << std::endl;
    }
#endif

    // 构建 cur_pattern_vid 位置的 unit_extend_size
    // 应用 restrictions 中的限制
    prepare_v_storage<config, cur_pattern_vid>(*device_context, prefix_storages,
                                               vertex_storages);

    auto &v_storage = vertex_storages[cur_pattern_vid];

    // 这个函数构建 cur_pattern_vid 位置的 unit_extend_sum
    gpuErrchk(cub::DeviceScan::InclusiveSum(
        d_prefix_temp_storage, temp_storage_bytes, v_storage.unit_extend_size,
        v_storage.unit_extend_sum, v_storage.num_units));

    // copy out sum 的结果，只需要最后一位
    // 注意这里是 Async，后面需要同步
    gpuErrchk(
        cudaMemcpyAsync(&extend_total_units,
                        v_storage.unit_extend_sum + v_storage.num_units - 1,
                        sizeof(VIndex_t), cudaMemcpyDeviceToHost));

#ifndef NDEBUG
    if (cur_pattern_vid < LOG_DEPTH) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Level " << cur_pattern_vid << ": leave prepare, time "
                  << duration.count() << " ms" << std::endl;
    }
#endif
}

template <Config config>
template <int cur_pattern_vid>
__host__ void Executor<config>::extend(int base_extend_unit_id,
                                       int num_extend_units) {
#ifndef NDEBUG
    auto time_start = std::chrono::high_resolution_clock::now();
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cout << "Level " << cur_pattern_vid << ": enter host extend"
                  << std::endl;
    }
#endif

    vertex_storages[cur_pattern_vid + 1].num_units = num_extend_units;

    extend_v_storage<config, cur_pattern_vid>(
        *device_context, prefix_storages, vertex_storages, base_extend_unit_id,
        num_extend_units);

    extend_p_storage<config, cur_pattern_vid>(
        *device_context, prefix_storages, vertex_storages, num_extend_units);

#ifndef NDEBUG
    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start);
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cout << "Level " << cur_pattern_vid << ": leave host extend"
                  << ", time " << duration.count() << " ms" << std::endl;
    }
#endif
}

template <Config config>
template <int cur_pattern_vid>
__host__ void Executor<config>::final_step() {
#ifndef NDEBUG
    auto time_start = std::chrono::high_resolution_clock::now();
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cout << "Level " << device_context->schedule_data.basic_vertexes
                  << ": enter final step" << std::endl;
    }
#endif

    // 这里是按照 IEP Info 的提示去计算出每一个 Unit 对应的答案，放到 d_ans 里面
    const auto &last_v_storage =
        vertex_storages[device_context->schedule_data.basic_vertexes - 1];

    get_iep_answer_kernel<config, cur_pattern_vid>
        <<<num_blocks, THREADS_PER_BLOCK>>>(*device_context, prefix_storages,
                                            vertex_storages, d_ans);

#ifndef NDEBUG
    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start);
    if (cur_pattern_vid < LOG_DEPTH) {
        std::cout << "Level " << device_context->schedule_data.basic_vertexes
                  << ": leave final step, time " << duration.count() << " ms"
                  << std::endl;
    }
#endif
}
}  // namespace Engine

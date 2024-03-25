#pragma once
#include <array>
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
    LevelStorage<config> *storages;
    unsigned long long ans;

    // CPU端的搜索
    template <int depth>
    __host__ void search(const DeviceContext<config> &context);

    template <int depth>
    __host__ void extend(const DeviceContext<config> &context) {
#ifndef NDEBUG
        std::cerr << "Enter Host Extend at level " << depth << std::endl;
#endif

        LevelStorage<config> &last = storages[depth];
        LevelStorage<config> &cur = storages[depth + 1];

        int start_index = 0;

        // 这个 For Each 是有序的，不是并行的。
        // 这个函数是有状态的！
        last.enumerate_unit([&](const StorageUnit<config> &unit) {
            VIndex_t num_vertexes;
            gpuErrchk(cudaMemcpy(&num_vertexes, &unit.vertex_set_size,
                                 sizeof(VIndex_t), cudaMemcpyDeviceToHost));

            // next level is full!
            if (start_index + num_vertexes >
                LevelStorage<config>::NUMS_STORAGE_UNIT) {
                return false;
            }

#ifndef NDEBUG
            std::cerr << "Extend Storage Unit, next_level start_index: "
                      << start_index << ", number_vertex: " << num_vertexes
                      << std::endl;
#endif
            int cur_prefix_id = depth + 1;
            auto next_level_units = cur.units();

            extend_storage_unit<config, depth>
                <<<num_blocks, THREADS_PER_BLOCK>>>(
                    context, unit, cur_prefix_id, next_level_units,
                    start_index);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());

            start_index += num_vertexes;

            return true;
        });

        cur._allocated_storage_units = start_index;

#ifndef DNEBUG
        std::cerr << "Leave Host Extend at level" << depth << std::endl;
#endif
    }

    // CPU 端结算答案
    __host__ void final_step_kernel(const DeviceContext<config> &context);

    __host__ void final_step(const DeviceContext<config> &context) {
        int depth = context.schedule_data.total_prefix_num - 1;

#ifndef NDEBUG
        std::cerr << "Enter Final Step, depth " << depth << std::endl;
#endif
        LevelStorage<config> &last = storages[depth];

        last.enumerate_unit([&](const StorageUnit<config> &unit) {
            VIndex_t num_vertexes;
            gpuErrchk(cudaMemcpy(&num_vertexes, &unit.vertex_set_size,
                                 sizeof(VIndex_t), cudaMemcpyDeviceToHost));

            this->ans += num_vertexes;

            return true;
        });
    }

  public:
    __host__ Executor(DeviceType device_type) : _device_type(device_type) {
#ifndef NDEBUG
        std::cerr << "Executor Constructor, ";
        std::cerr << "Device Type: " << device_type << std::endl;
#endif
        if (device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaMalloc(&storages,
                                 sizeof(LevelStorage<config>) * MAX_DEPTH));
        } else if (device_type == DeviceType::CPU_DEVICE) {
            storages = new LevelStorage<config>[MAX_DEPTH];
        } else {
            assert(false);
        }
    }

    __host__ ~Executor() {
        if (_device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaFree(&storages));
        } else if (_device_type == DeviceType::CPU_DEVICE) {
            delete[] storages;
        } else {
            assert(false);
        }
    }

    __host__ unsigned long long perform_search(
        const DeviceContext<config> &context) {
        // 重置答案
        this->ans = 0;
        // 在这里做第一层，把 Neighborhood 填进去
        VIndex_t start_vertex = 0;
        VIndex_t *size_this_time_dev;
        VIndex_t size_this_time;
        gpuErrchk(cudaMalloc(&size_this_time_dev, sizeof(VIndex_t)));
        while (true) {
            first_layer_kernel<config><<<1, 1>>>(
                context, storages[0].units(), start_vertex, size_this_time_dev);

            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());

            gpuErrchk(cudaMemcpy(&size_this_time, size_this_time_dev,
                                 sizeof(VIndex_t), cudaMemcpyDeviceToHost));

            if (size_this_time == 0) {
                break;
            }

            storages[0].clear();
            storages[0]._allocated_storage_units = size_this_time;

            start_vertex += size_this_time;

            search<0>(context);
        }
        // 从第 0 个 prefix 开始搜索
        return this->ans;
    }
};

template <Config config>
__host__ void Executor<config>::final_step_kernel(
    const DeviceContext<config> &context) {}

template <Config config>
template <int depth>
__host__ void Executor<config>::search(const DeviceContext<config> &context) {
#ifndef NDEBUG
    std::cerr << "search, enter level " << depth << std::endl;
#endif

    // 如果已经到达了终点，进入最终的处理
    if (depth == context.schedule_data.total_prefix_num - 1) {
#ifndef NDEBUG
        std::cerr << "search, final step" << std::endl;
#endif
        final_step(context);
        return;
    }

    const LevelStorage<config> &last = storages[depth];
    LevelStorage<config> &current = storages[depth + 1];

    while (true) {
        // 直到上一层全部被拓展完完成
        if (last.extend_finished()) {
#ifndef NDEBUG
            std::cerr << "search, level " << depth << " finished" << std::endl;
#endif
            break;
        }

        // 清除下一层的存储
        current.clear();
        // 重新分配存储
        current.reallocate();

        // 从当前进度进行当前一次拓展
        extend<depth>(context);

        usleep(10000);

        // 递归进行搜素
        // 这句 if constexpr 不能缺少，否则会导致编译器无限递归
        // 不用特化的原因是，模板类中，成员函数无法单独特化
        if constexpr (depth + 1 < MAX_DEPTH) {
            search<depth + 1>(context);
        }
    }
}

}  // namespace Engine

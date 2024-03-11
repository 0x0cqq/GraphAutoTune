#pragma once
#include <array>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "core/graph.cuh"
#include "core/schedule.hpp"
#include "core/types.hpp"
#include "core/vertex_set.cuh"
#include "engine/storage.cuh"
#include "engine/worker.cuh"
#include "infra/graph_backend.cuh"

namespace Engine {

// GPU 上，整个 Device 所需要的所有的其他的东西，都一起打包进来，例如 Schedule
// 等
struct GPUDeviceContext {
    Core::Schedule schedule;
};

template <Config config>
class Executor {
  private:
    constexpr static int MAX_DEPTH = 10;
    using GraphBackend = GraphBackendTypeDispatcher<config>::type;

    // 提供图数据访问的后端
    GraphBackend graph_backend;
    // 其他需要传递进来的指针等
    GPUDeviceContext deviceContext;
    // 搜索过程中每层所需要的临时存储
    LevelStorage<config> storages[MAX_DEPTH];

    // 扩展
    template <int depth>
    __device__ void extend();

    // 搜索
    template <int depth>
    __device__ void search();

  public:
    // 没有模板的搜索函数。用于外部调用
    __device__ void perform_search() {
        // 先建第一层
        search<1>();
    }
};

// declare search

template <Config config>
template <int depth>
__device__ void Executor<config>::extend() {
    extern __shared__ WorkerInfo workerInfos[];
    const LevelStorage<config> &last = storages[depth - 1];
    LevelStorage<config> &current = storages[depth];
    WorkerInfo &workerInfo = workerInfos[depth];
    const int wid = threadIdx.x / THREADS_PER_BLOCK;
    const int lid = threadIdx.x % THREADS_PER_BLOCK;
    // 在 GPU 上 Extend，每个 Warp 作为一个 Worker.

#ifndef NDEBUG
    if (wid == 0 && lid == 0) {
        printf("extend: %d\n", depth);
    }
#endif
}

template <Config config>
template <int depth>
__device__ void Executor<config>::search() {
    const int wid = threadIdx.x / THREADS_PER_BLOCK;
    const int lid = threadIdx.x % THREADS_PER_BLOCK;
#ifndef NDEBUG
    if (wid == 0 && lid == 0) {
        printf("search: %d\n", depth);
    }
#endif
    // 维护拓展进度

    while (true) {
        // 直到上一层全部被拓展完完成

        // 从当前进度进行当前一次拓展
        extend<depth>();

        // 递归进行搜素
        // 这句不能缺少，否则会导致编译器无限递归
        // 不用特化的原因是，成员函数无法单独特化
        if constexpr (depth < MAX_DEPTH) {
            search<depth + 1>();
        }
    }
}

}  // namespace Engine

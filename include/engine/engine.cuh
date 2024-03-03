#pragma once
#include <array>

#include "configs/gpu_consts.cuh"
#include "core/graph.cuh"
#include "core/types.hpp"
#include "core/vertex_set.cuh"
#include "engine/storage.cuh"
#include "engine/worker.cuh"
#include "infra/graph_backend.cuh"

namespace Engine {

// GPU 上，整个 Device
struct GPUDeviceContext {};

template <Config config>
class Executor {
  public:
    using GraphBackend = GraphBackendTypeDispatcher<config>::type;

    GraphBackend graph_backend;
    GPUDeviceContext deviceContext;
    LevelStorage<config> storages[MAX_DEPTH];

    constexpr static int MAX_DEPTH = 10;
    template <int depth>
    __device__ void extend() {
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

    template <int depth>
    __device__ void search();

    __device__ void perform_search() {
        // 先建第一层
        search<1>();
    }
};

// declare search

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
    extend<depth>();
    if constexpr (depth < MAX_DEPTH) {
        search<depth + 1>();
    }
}

}  // namespace Engine

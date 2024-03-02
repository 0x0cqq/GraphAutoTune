#pragma once
#include <array>

#include "configs/gpu_consts.cuh"
#include "core/types.hpp"
#include "core/vertex_set.cuh"
#include "engine/storage.cuh"
#include "engine/worker.cuh"

namespace Engine {

// GPU 上，整个 Device
struct GPUDeviceContext {};

template <Config config>
class Executor {
  public:
    using VertexSetImpl = Core::VertexSetTypeDispatcher<config>::type;
    GPUDeviceContext deviceContext;
    LevelStorage<VertexSetImpl> storages[MAX_DEPTH];

    constexpr static int MAX_DEPTH = 10;
    template <int depth>
    __device__ void extend() {
        extern __shared__ WorkerInfo workerInfos[];
        const LevelStorage<VertexSetImpl> &last = storages[depth - 1];
        LevelStorage<VertexSetImpl> &current = storages[depth];
        WorkerInfo &workerInfo = workerInfos[depth];
        const int wid = threadIdx.x / THREADS_PER_BLOCK;
        const int lid = threadIdx.x % THREADS_PER_BLOCK;
        // 在 GPU 上 Extend，每个 Warp 作为一个 Worker.
    }

    template <int depth>
    __device__ void search() {
        extend<depth>();
        search<depth + 1>(storages);
    }

    template <>
    __device__ void search<MAX_DEPTH>() {
        // 递归边界
    }

    __global__ void perform_search() {
        // 先建第一层
        search<1>();
    }
};

}  // namespace Engine

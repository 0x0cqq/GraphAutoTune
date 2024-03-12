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
    unsigned long long answer;
};

template <Config config>
class Executor {
  private:
    constexpr static int MAX_DEPTH = 10;
    using GraphBackend = GraphBackendTypeDispatcher<config>::type;

    // 提供图数据访问的后端
    GraphBackend graph_backend;
    // 其他需要传递进来的指针等
    GPUDeviceContext device_context;
    // 搜索过程中每层所需要的临时存储
    LevelStorage<config> storages[MAX_DEPTH];

    // 扩展
    template <int depth>
    __device__ void extend();

    // 搜索
    template <int depth>
    __device__ void search();

    // 结算答案
    __device__ void final_step();

  public:
    // 没有模板的搜索函数。用于外部调用
    __device__ void perform_search() {
        extern __shared__ WorkerInfo workerInfos[];
        const int wid = threadIdx.x / THREADS_PER_WARP;
        const int lid = threadIdx.x % THREADS_PER_WARP;

        // 对 Worker 进行初始化
        if (lid == 0) {
            workerInfos[wid].clear();
        }

        // 从第 0 个 prefix 开始搜索
        search<0>();

        // 归结到设备的答案上
        if (lid == 0) {
            atomicAdd(&device_context.answer, workerInfos[wid].local_answer);
        }
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
    const int wid = threadIdx.x / THREADS_PER_WARP;
    const int lid = threadIdx.x % THREADS_PER_WARP;
    // 在 GPU 上 Extend，每个 Warp 作为一个 Worker.

#ifndef NDEBUG
    if (wid == 0 && lid == 0) {
        printf("extend: %d\n", depth);
    }
#endif
    // 并行地枚举上一层的所有的 vertex set 的所有的节点

    // 先进行一次代价较低的计算（每个点O(1)），得到每个点产生的 Vertex
    // 所需要的存储大小

    // 然后再分配空间

    // 然后下面的就可以完美并行了

    // loop_vertex 也就是这个节点，每一个节点可以产生一个新的 vertex
    // set，填在下一层

    // 根据 prefix 的 father，在 storages 的某一层的某一个位置，找到 father
    // dependency 的 vertex set

    // 两者求交得到新的 Vertex Set

    // 如果新的 Vertex Set 不为空，那么就放到下一层的 storages 中
}

template <Config config>
__device__ void Executor<config>::final_step() {
    extern __shared__ WorkerInfo workerInfos[];
    const int wid = threadIdx.x / THREADS_PER_WARP;
    const int lid = threadIdx.x % THREADS_PER_WARP;
    if (lid == 0) {
        // TODO: This need to be fill
        workerInfos[wid].local_answer += 1;
    }
}

template <Config config>
template <int depth>
__device__ void Executor<config>::search() {
    const int wid = threadIdx.x / THREADS_PER_WARP;
    const int lid = threadIdx.x % THREADS_PER_WARP;
#ifndef NDEBUG
    if (wid == 0 && lid == 0) {
        printf("search: %d\n", depth);
    }
#endif

    // 如果已经到达了终点，进入最终的处理
    if (depth == device_context.schedule.total_prefix_num - 1) {
        final_step();
        return;
    }

    const LevelStorage<config> &last = storages[depth - 1];
    LevelStorage<config> &current = storages[depth];

    while (true) {
        // 直到上一层全部被拓展完完成，通过 last 的 scanned vertex 来判断
        if (last.extend_finished()) {
            break;
        }

        // 清除下一层的存储
        current.clear();
        // 从当前进度进行当前一次拓展
        extend<depth>();

        // 递归进行搜素
        // 这句 if constexpr 不能缺少，否则会导致编译器无限递归
        // 不用特化的原因是，模板类中，成员函数无法单独特化
        if constexpr (depth + 1 < MAX_DEPTH) {
            search<depth + 1>();
        }
    }
}

}  // namespace Engine

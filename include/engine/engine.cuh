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

namespace Engine {

// 整个 Device 所需要的所有变量一起打包进来，如 ScheduleData 等。
// 这里面的变量应当是不会改变的，我们将会将其整体放进 constant memory 中。
// 通过直接给 Kernel 函数传参
// 之后考虑通过元编程传递一部分。
template <Config config>
struct DeviceContext {
    using GraphBackend = GraphBackendTypeDispatcher<config>::type;
    // 图挖掘的 Schedule
    Core::ScheduleData schedule_data;
    // 提供图数据访问的后端
    GraphBackend graph_backend;

    __host__ DeviceContext(const Core::Schedule &_schedule,
                           const GraphBackend &_graph_backend)
        : schedule_data(_schedule), graph_backend(_graph_backend) {}

    __host__ void to_device() {
        schedule_data.to_device();
        graph_backend.to_device();
    }
};

template <Config config>
class Executor {
  private:
    using VertexSet = VertexSetTypeDispatcher<config>::type;

    // 搜索过程中每层所需要的临时存储
    LevelStorage<config> *storages;

    // 扩展
    template <int depth>
    __device__ void extend(const DeviceContext<config> &context);

    // 搜索
    template <int depth>
    __device__ void search(const DeviceContext<config> &context);

    // 结算答案
    __device__ void final_step(const DeviceContext<config> &context);

  public:
    __host__ Executor(DeviceType device_type) {
        if (device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaMalloc(&storages,
                                 sizeof(LevelStorage<config>) * MAX_DEPTH));
        } else if (device_type == DeviceType::CPU_DEVICE) {
            storages = new LevelStorage<config>[MAX_DEPTH];
        } else {
            assert(false);
        }
    }

    // 没有模板的搜索函数。用于外部 global kernel 调用
    __device__ void perform_search(unsigned long long *ans,
                                   const DeviceContext<config> &context) {
        extern __shared__ WorkerInfo workerInfos[];
        const int wid = threadIdx.x / THREADS_PER_WARP;
        const int lid = threadIdx.x % THREADS_PER_WARP;

        // 对 Worker 进行初始化
        if (lid == 0) {
            workerInfos[wid].clear();
        }

        // 从第 0 个 prefix 开始搜索
        search<0>(context);

        // 归结到设备的答案上
        if (lid == 0) {
            atomicAdd(ans, workerInfos[wid].local_answer);
        }
    }
};

// declare search
template <Config config>
template <int depth>
__device__ void Executor<config>::extend(const DeviceContext<config> &context) {
    // extern __shared__ WorkerInfo workerInfos[];
    const LevelStorage<config> &last = storages[depth - 1];
    LevelStorage<config> &current = storages[depth];
    // WorkerInfo &workerInfo = workerInfos[depth];
    const int wid = threadIdx.x / THREADS_PER_WARP;
    const int lid = threadIdx.x % THREADS_PER_WARP;
    // 在 GPU 上 Extend，每个 Warp 作为一个 Worker.

    const int prefix_id = depth;

#ifndef NDEBUG
    if (wid == 0 && lid == 0) {
        printf("extend: %d\n", depth);
    }
#endif
    // 这里先编写一个最简单的版本：每个 Vertex Set 的空间是固定的

    // 并行地枚举上一层的所有的 vertex set 的所有的节点
    // 遍历 Vertex Set
    for (int i = last.cur_storage_unit(); i < last.allocated_storage_units();
         i++) {
        const auto &storage_unit = last.storage_unit(i);
        int size = storage_unit.vertex_set.size();
        // 遍历 Set 中的 Vertex (每个 Warp 分到一个节点)
        storage_unit.vertex_set.foreach_vertex([&](VIndex_t new_v) {
            // 扩展操作
            VIndex_t *neighbors = context.graph_backend.get_neigh(new_v);
            VIndex_t neighbors_cnt = context.graph_backend.get_neigh_cnt(new_v);
            // 构建 Vertex Set: Neighbor Vertex Set
            VertexSet new_vertex_set;
            new_vertex_set.init(neighbors, neighbors_cnt);

            // 找到 father 的 Vertex Set
            // 当前正在处理第 prefix_id 个 prefix
            int father_index = context.schedule_data.prefix_fathers[prefix_id];
            const VertexSet &father_vertex_set =
                storage_unit.fathers[father_index]->vertex_set;

            // 找到下一层的 Vertex Set
            // TODO: index_new 还不知道怎么获得。要和 allocate
            // 一起获得？allocate 要和 storage unit 耦合到一起吧。
            int index_new = 1;
            VertexSet &next_vertex_set =
                current.storage_unit(index_new).vertex_set;
            VIndex_t *space = current.allocate();
            next_vertex_set.init_empty(
                space, neighbors_cnt);  // 不会比 neighbors_cnt 大 //
                                        // 之后考虑优化掉这个 size 参数
            next_vertex_set.intersect(father_vertex_set, new_vertex_set);
        });
    }

    // 分配空间：进行一次前缀和。

    // 扩展：相互独立的。

    // for loop_vertex in last.vertexes:
    // 这个循环怎么写呢？如果 storage Unit 的 space
    // 不能完整确定，这个循环是两层的 外层是 storage unit，内层才是 vertex 每个
    // vertex 要分给一个不同的 warp，所以这里同步只能靠 atomic。 两个变量的
    // atomic 是很难搞的。 ~1000 warp

    // 还有一个问题，就是怎么从 index 反查属于哪个 storage unit。
    // block -> storage unit?
    // 如果一个 block 16K 的数，也就是 64KB (fit 一下 L2 Cache？)
    // 那么总共 16GB 的存储，需要 1M / 4 = 256K 个 block，可以维护一个反查表？

    // loop_vertex 也就是这个节点，每一个节点可以产生一个新的 vertex set
    // vset_1 = neighbor(loop vertex)

    // 根据 prefix 的 father，在 storages
    // 的某一层的某一个位置[这个是对于所有的节点都是一致的] 找到 father
    // dependency set 对应的 vertex set father_vertex_set =
    // storages[prefix.father][father_dependency_set_id]
    //
    // 100 个 last 里面的 vertex，实际可能需要的 vertex set 只有几个
    // 大大减小读存储压力？
    // 但是会不会带来内存冲突？每个 Warp 读的都是同样的内存，很多个 Warp
    // 读的也很可能是同样的内存
    // L1表示一级缓存，每个SM都有自己L1，但是L2是所有SM公用的，除了L1缓存外，还有只读缓存和常量缓存。

    // 两者求交得到新的 Vertex Set
    // new_vertex_set = vset_1.intersect(father_vertex_set)
    // current[xxx.id] = new_vertex_set

    // 如果新的 Vertex Set 不为空，那么就放到下一层的 storages 中
}

template <Config config>
__device__ void Executor<config>::final_step(
    const DeviceContext<config> &context) {
    extern __shared__ WorkerInfo workerInfos[];
    const int wid = threadIdx.x / THREADS_PER_WARP;
    const int lid = threadIdx.x % THREADS_PER_WARP;
    if (lid == 0) {
        // TODO: 目前仅为占位符
        workerInfos[wid].local_answer += 1;
    }
}

template <Config config>
template <int depth>
__device__ void Executor<config>::search(const DeviceContext<config> &context) {
    const int wid = threadIdx.x / THREADS_PER_WARP;
    const int lid = threadIdx.x % THREADS_PER_WARP;
#ifndef NDEBUG
    if (wid == 0 && lid == 0) {
        printf("search: %d\n", depth);
    }
#endif

    // 如果已经到达了终点，进入最终的处理
    if (depth == context.schedule_data.total_prefix_num - 1) {
        final_step(context);
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
        extend<depth>(context);

        // 递归进行搜素
        // 这句 if constexpr 不能缺少，否则会导致编译器无限递归
        // 不用特化的原因是，模板类中，成员函数无法单独特化
        if constexpr (depth + 1 < MAX_DEPTH) {
            search<depth + 1>(context);
        }
    }
}

// kernel

// 注意， Engine 总共不能超过 32KB，否则无法放到 constant memory
// 中，我们就只能再传指针。
template <Config config>
__global__ void pattern_matching_kernel(Executor<config> engine,
                                        DeviceContext<config> context,
                                        unsigned long long *ans) {
    engine.perform_search(ans, context);
}

// 封装了一下
template <Config config>
unsigned long long pattern_matching(Executor<config> &engine,
                                    DeviceContext<config> &context) {
    unsigned long long *ans;
    gpuErrchk(cudaMallocManaged(&ans, sizeof(unsigned long long)));
    *ans = 0ull;

    pattern_matching_kernel<config>
        <<<num_blocks, THREADS_PER_BLOCK,
           sizeof(WorkerInfo) * MAX_DEPTH * WARPS_PER_BLOCK>>>(engine, context,
                                                               ans);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    unsigned long long ret = *ans;
    gpuErrchk(cudaFree(ans));
    return ret;
}

}  // namespace Engine

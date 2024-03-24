#pragma once

#include "engine/context.cuh"
#include "engine/storage.cuh"

namespace Engine {

template <Config config, int depth>
__global__ void extend_storage_unit(DeviceContext<config> context,
                                    const StorageUnit<config> &unit,
                                    int cur_prefix_id,
                                    StorageUnit<config> *next_level_units,
                                    int start_index) {
    using VertexSet = VertexSetTypeDispatcher<config>::type;
    __shared__ VertexSet new_vertex_sets[WARPS_PER_BLOCK];

    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    int father_prefix_id = context.schedule_data.prefix_fathers[cur_prefix_id];
    const VertexSet &father_vertex_set =
        unit.fathers[father_prefix_id]->vertex_set;
    if (lane_id == 0) {
        printf("warp_id: %d", warp_id);
        printf("cur_prefix_id: %d\n", cur_prefix_id);
        printf("father_prefix_id: %d set:%p\n", father_prefix_id,
               &father_vertex_set);
        printf("father_vertex_set.size(): %d\n", father_vertex_set.size());
    }

    unit.vertex_set.foreach_vertex([&](VIndex_t new_v, size_t index) {
        // 扩展操作
        // 构建 邻居Vertex Set
        // 这两步应该提走，单独封装出去
        VIndex_t *neighbors = context.graph_backend.get_neigh(new_v);
        VIndex_t neighbors_cnt = context.graph_backend.get_neigh_cnt(new_v);
        VertexSet &new_vertex_set = new_vertex_sets[warp_id];
        new_vertex_set.init(neighbors, neighbors_cnt);

        // 找到 father 的 Vertex Set
        // 当前正在处理第 prefix_id 个 prefix

        // 找到下一层的 Storage Unit
        int next_level_index = start_index + index;
        StorageUnit<config> &next_unit = next_level_units[next_level_index];
        auto &next_vertex_set = next_unit.vertex_set;

        next_vertex_set.intersect(father_vertex_set, new_vertex_set);
        next_unit.vertex_set_size = next_vertex_set.size();
        // // 更新 father 链
        for (int i = 0; i < cur_prefix_id; i++) {
            next_unit.fathers[i] = unit.fathers[i];
        }
        next_unit.fathers[cur_prefix_id] = &next_unit;

        // 更新 Unordered Vertex Set
        next_unit.subtraction_set.copy(unit.subtraction_set);
        next_unit.subtraction_set.set<depth>(new_v);
    });
}

template <Config config>
__global__ void first_layer_kernel(DeviceContext<config> context,
                                   StorageUnit<config> *units,
                                   VIndex_t start_vertex_index,
                                   VIndex_t *size_this_time) {
    VIndex_t v_cnt = context.graph_backend.v_cnt();
    VIndex_t size_vertex = min(LevelStorage<config>::NUMS_STORAGE_UNIT,
                               v_cnt - start_vertex_index);
#ifndef NDEBUG
    printf("v_cnt: %d size_vertex: %d\n", v_cnt, size_vertex);
#endif
    for (VIndex_t i = 0; i < size_vertex; i++) {
        VIndex_t vertex_index = start_vertex_index + i;
        VIndex_t *neighbors = context.graph_backend.get_neigh(vertex_index);
        VIndex_t neighbors_cnt =
            context.graph_backend.get_neigh_cnt(vertex_index);
        using VertexSet = VertexSetTypeDispatcher<config>::type;
        units[i].vertex_set.init(neighbors, neighbors_cnt);
        units[i].vertex_set_size = neighbors_cnt;
        units[i].subtraction_set.set<0>(vertex_index);

        // 存一下自己
        units[i].fathers[0] = &units[i];
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *size_this_time = size_vertex;
    }
}

}  // namespace Engine
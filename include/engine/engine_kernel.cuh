#pragma once

#include "engine/context.cuh"
#include "engine/storage.cuh"

namespace Engine {

template <Config config, int depth>
__global__ void extend_storage_unit(const DeviceContext<config> &context,
                                    const StorageUnit<config> &unit,
                                    int cur_prefix_id,
                                    StorageUnit<config> *next_level_units,
                                    int start_index) {
    using VertexSet = VertexSetTypeDispatcher<config>::type;
    int father_prefix_id = context.schedule_data.prefix_fathers[cur_prefix_id];
    const VertexSet &father_vertex_set =
        unit.fathers[father_prefix_id]->vertex_set;

    unit.vertex_set.foreach_vertex([&](VIndex_t new_v, size_t index) {
        // 扩展操作
        // 构建 邻居Vertex Set
        // 这两步应该提走，单独封装出去
        VIndex_t *neighbors = context.graph_backend.get_neigh(new_v);
        VIndex_t neighbors_cnt = context.graph_backend.get_neigh_cnt(new_v);
        VertexSet new_vertex_set;
        new_vertex_set.init(neighbors, neighbors_cnt);

        // 找到 father 的 Vertex Set
        // 当前正在处理第 prefix_id 个 prefix

        // 找到下一层的 Storage Unit
        int next_level_index = start_index + index;
        StorageUnit<config> &next_unit = next_level_units[next_level_index];
        auto &next_vertex_set = next_unit.vertex_set;
        next_vertex_set.intersect(father_vertex_set, new_vertex_set);
        next_unit.vertex_set_size = next_vertex_set.size();
        // 更新 father 链
        for (int i = 0; i < cur_prefix_id; i++) {
            next_unit.fathers[i] = unit.fathers[i];
        }
        next_unit.fathers[cur_prefix_id] = &unit;

        // 更新 Unordered Vertex Set
        next_unit.subtraction_set.copy(unit.subtraction_set);
        next_unit.subtraction_set.set<depth>(new_v);
    });
}

}  // namespace Engine
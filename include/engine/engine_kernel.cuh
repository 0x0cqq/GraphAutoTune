#pragma once

#include "engine/context.cuh"
#include "engine/storage.cuh"

namespace Engine {

// 这个 Kernel 是一个一个遍历所有的点，然后构建第一层的 Storage
// 是否可以把这个地方的 遍历 Vertex Set 抽象成遍历一个
// Function，然后在第一层这个 function 就是 identity
// 在其他层可能就是一个很复杂的 function
// 这个function 是独立的

// space 是单独开的，需要再分给 vertex set 里面的 pointer
template <Config config>
__global__ void set_vertex_set_space(PrefixStorage<config> p_storage) {
    const int wid = threadIdx.x / 32;
    const int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;

    for (int base = 0; base < NUMS_UNIT; base += num_total_warps) {
        VIndex_t uid = base + global_wid;
        if (uid >= NUMS_UNIT) continue;
        p_storage.vertex_set[uid].init_empty(
            p_storage.space + uid * MAX_SET_SIZE, MAX_SET_SIZE);
    }
}

// 传进来的是第一层的 prefix storage 和 vertex storage
// 注意，第一层的 prefix storage 和 vertex storage 都一定只有一个
// 也就是[0] 这个 Prefix
template <Config config>
__global__ void first_extend_kernel(DeviceContext<config> context,
                                    PrefixStorage<config> p_storage,
                                    VertexStorage<config> v_storage,
                                    VIndex_t start_vid, VIndex_t end_vid) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    for (VIndex_t base = start_vid; base < end_vid; base += num_total_warps) {
        VIndex_t vid = base + global_warp_id;
        if (vid >= end_vid) continue;
        VIndex_t uid = vid - start_vid;

        VIndex_t *neighbors = context.graph_backend.get_neigh(vid);
        VIndex_t neighbors_cnt = context.graph_backend.get_neigh_cnt(vid);
        p_storage.vertex_set[uid].init_copy(neighbors, neighbors_cnt);
        if (lane_id == 0) {
            v_storage.subtraction_set[uid].set<0>(vid);
        }
    }
}

// 找到在 v_storages[curr_pattern_vid][uid] 在 father_pattern_id 深度的 father
// 可以每个 thread（而非 Warp） 找一个不同的 uid
template <Config config>
__device__ int find_father(int cur_pattern_vid, int father_pattern_vid,
                           VertexStorages<config> &vertex_storages, int uid) {
    int cur_uid = uid;
    for (int i = cur_pattern_vid; i > father_pattern_vid; --i) {
        cur_uid = vertex_storages[i].previous_index[cur_uid];
    }
    return cur_uid;
}

// 我们希望知道某个 prefix id 是属于哪个 vertex，其实就是最后一个vertex 是什么
template <Config config>
__device__ int from_prefix_id_to_vertex_id(const DeviceContext<config> &context,
                                           int prefix_id) {
    const auto &prefix = context.schedule_data.prefixes[prefix_id];
    return prefix.data[prefix.depth - 1];
}

template <Config config, int cur_pattern_vid>
__device__ int find_loop_set_uid(const DeviceContext<config> &context,
                                 VertexStorages<config> &v_storages,
                                 int loop_set_prefix_id, int uid) {
    int loop_set_vertex_id =
        from_prefix_id_to_vertex_id(context, loop_set_prefix_id);
    return find_father(cur_pattern_vid, loop_set_vertex_id, v_storages, uid);
}

template <Config config, int cur_pattern_vid>
__device__ int find_father_uid(const DeviceContext<config> &context,
                               VertexStorages<config> &v_storages,
                               int father_prefix_id, int uid) {
    int father_vertex_id =
        from_prefix_id_to_vertex_id(context, father_prefix_id);
    return find_father(cur_pattern_vid, father_vertex_id, v_storages, uid);
}

// 已经扩展了 [0, cur_pattern_vid) 的 prefix，现在需要扩展包含 cur_pattern_vid
// 的 prefix。
// 我们将会扩展 [v_prefix_s[cur_pattern_vid], v_prefix_s[cur_pattern_vid + 1])
// 的 Prefix。
// [start_uid, end_uid)
template <Config config, int cur_pattern_vid>
__global__ void extend_p_storage(const DeviceContext<config> context,
                                 PrefixStorages<config> p_storages,
                                 VertexStorages<config> v_storages,
                                 int start_uid, int end_uid) {
    using VertexSet = VertexSetTypeDispatcher<config>::type;
    __shared__ VertexSet new_vertex_sets[WARPS_PER_BLOCK];

    const int loop_set_prefix_id =
        context.schedule_data.loop_set_prefix_id[cur_pattern_vid];
    const auto &loop_set_storage = p_storages[loop_set_prefix_id];

    const auto &cur_v_storage = v_storages[cur_pattern_vid - 1];
    const auto &next_v_storage = v_storages[cur_pattern_vid];

    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int global_wid = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (global_wid == 0 && lane_id == 0) {
        printf("loop_set_prefix_id: %d\n", loop_set_prefix_id);
    }

    // 获取 start 和 end 的 prefix id
    const int start_prefix_id =
        context.schedule_data.vertex_prefix_start[cur_pattern_vid];
    const int end_prefix_id =
        context.schedule_data.vertex_prefix_start[cur_pattern_vid + 1];

    // 需要遍历 v in loop_set_storage ，v 就是我们选的 cur_pattern_vid 节点
    // 我们会需要它的 N(v)

    // 接下来 是要 forall prefixes 的
    // 如果 father_prefix_id != 0
    // 对于每一个 v （对于 v 属于的 unit），我们需要找到 father_prefix_id
    // 那一层的 p_storage 中 这个 prefix 的 father 对应的 unit id

    // 但是这个 Unit id 不通过 p_storage 来找，通过 v_father =
    // father_prefix_id.last 通过 v_storage 往上一层层捋 father uid。

    // 找到 uid 之后，提取出来 vertex set

    // 在目的地的 uid 里面相交。

    int end_extend_unit_id = cur_v_storage.unit_extend_sum[end_uid - 1];
    int start_extend_unit_id =
        start_uid == 0 ? 0 : cur_v_storage.unit_extend_sum[start_uid - 1];

    int num_extend_units = end_extend_unit_id - start_extend_unit_id;

    for (int base = 0; base < num_extend_units; base += num_total_warps) {
        int next_extend_uid = base + global_wid;
        if (next_extend_uid >= num_extend_units) continue;
        // 反推到上一层的 Unit 和 Vertex，群体二分查找
        int l = start_uid, r = end_uid - 1;  // [l, r] 闭区间
        int target_value = start_extend_unit_id + next_extend_uid;
        while (l != r) {
            int mid = (l + r) / 2;
            int mid_value = cur_v_storage.unit_extend_sum[mid];

            if (mid_value <= target_value) {
                l = mid + 1;
            } else {  // mid_value > target_value
                r = mid;
            }
        }

        int cur_level_uid = l;
        int base_index = cur_level_uid == 0
                             ? 0
                             : cur_v_storage.unit_extend_sum[cur_level_uid - 1];
        int vertex_set_size = cur_v_storage.unit_extend_size[cur_level_uid];
        int vertex_index = target_value - base_index;

        int loop_set_uid = find_loop_set_uid<config, cur_pattern_vid>(
            context, v_storages, loop_set_prefix_id, cur_level_uid);

        if (lane_id == 0) {
            // printf(
            //     "target_value: %d "
            //     "cur_level_uid: %d vertex_index: %d base: %d size: %d\n",
            //     target_value, cur_level_uid, vertex_index, base_index,
            //     vertex_set_size);
            // // printf("pointer: %p\n",
            // //        loop_set_storage.vertex_set[loop_set_uid].data());
            assert(vertex_index >= 0);
            assert(vertex_index < vertex_set_size);
        }

        // vertex + index 获取 vertex 的接口
        VIndex_t v =
            loop_set_storage.vertex_set[loop_set_uid].get(vertex_index);

        // 构建邻居Vertex Set
        // 这两步应该提走，单独封装出去
        VIndex_t *neighbors = context.graph_backend.get_neigh(v);
        VIndex_t neighbors_cnt = context.graph_backend.get_neigh_cnt(v);
        VertexSet &new_vertex_set = new_vertex_sets[warp_id];
        new_vertex_set.init(neighbors, neighbors_cnt);

        // 构建一堆 p_storage
        for (int cur_prefix_id = start_prefix_id; cur_prefix_id < end_prefix_id;
             cur_prefix_id++) {
            // 找到下一层的 Storage Unit
            auto &next_vertex_set =
                p_storages[cur_prefix_id].vertex_set[next_extend_uid];

            // 找到 father 的 Vertex Set
            // 当前正在处理第 prefix_id 个 prefix
            int father_prefix_id =
                context.schedule_data.prefix_fathers[cur_prefix_id];
            if (father_prefix_id == -1) {
                next_vertex_set.init_copy(neighbors, neighbors_cnt);
            } else {
                const auto &father_prefix =
                    context.schedule_data.prefixes[father_prefix_id];

                int father_unit_id = find_father_uid<config, cur_pattern_vid>(
                    context, v_storages, father_prefix_id, cur_level_uid);
                const auto &father_vertex_set =
                    p_storages[father_prefix_id].vertex_set[father_unit_id];
                next_vertex_set.intersect(new_vertex_set, father_vertex_set);
            }
        }
    }
}

template <Config config, int cur_pattern_vid>
__global__ void extend_v_storage(const DeviceContext<config> context,
                                 PrefixStorages<config> prefix_storages,
                                 VertexStorages<config> vertex_storages) {
    const auto &v_storage = vertex_storages[cur_pattern_vid];
    const int num_units = v_storage.num_units;
    const int bid = blockIdx.x, tid = threadIdx.x;
    const int global_tid = bid * blockDim.x + tid;
    const int num_threads = gridDim.x * blockDim.x;
    for (int base = 0; base < num_units; base += num_threads) {
        int uid = base + global_tid;
        if (uid >= num_units) continue;
        // 构建 loop_set_unit 的 size
        int loop_set_prefix_id =
            context.schedule_data.loop_set_prefix_id[cur_pattern_vid + 1];
        // 要改的是 v，但是是给 v + 1 用的
        int loop_set_uid = find_loop_set_uid<config, cur_pattern_vid>(
            context, vertex_storages, loop_set_prefix_id, uid);
        v_storage.unit_extend_size[uid] =
            prefix_storages[loop_set_prefix_id].vertex_set[loop_set_uid].size();
    }
}

template <Config config>
__global__ void get_next_unit(int current_unit, int *next_unit,
                              int *next_total_units, int num_units,
                              VertexStorage<config> v_storage) {
    VIndex_t current_unit_size =
        current_unit == 0 ? 0 : v_storage.unit_extend_sum[current_unit - 1];
    VIndex_t next_size = current_unit_size + num_units;
    *next_unit = current_unit;
    while (*next_unit < v_storage.num_units &&
           v_storage.unit_extend_sum[*next_unit] < next_size) {
        (*next_unit)++;
    }
    *next_total_units =
        v_storage.unit_extend_sum[*next_unit - 1] - current_unit_size;
}

}  // namespace Engine
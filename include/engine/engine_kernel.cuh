#pragma once

#include <cooperative_groups.h>

#include "consts/project_consts.hpp"
#include "engine/context.cuh"
#include "engine/storage.cuh"
#include "utils/utils.hpp"

namespace cg = cooperative_groups;

namespace Engine {

int counter = 0;

// 这个 Kernel 是一个一个遍历所有的点，然后构建第一层的 Storage
// 是否可以把这个地方的 遍历 Vertex Set 抽象成遍历一个
// Function，然后在第一层这个 function 就是 identity
// 在其他层可能就是一个很复杂的 function
// 这个 function 是独立的

// space 是单独开的，需要再分给 vertex set 里面的 pointer
template <Config config>
__global__ void set_vertex_set_space_kernel(PrefixStorage<config> p_storage,
                                            int num_units, int set_size) {
    auto grid = cg::this_grid();

    const int global_tid = grid.thread_rank();
    const int num_threads = grid.num_threads();

    for (int uid = global_tid; uid < num_units; uid += num_threads) {
        p_storage.vertex_set[uid].init_empty(p_storage.space + uid * set_size,
                                             set_size);
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
    auto grid = cg::this_grid();

    const int global_tid = grid.thread_rank();
    const int num_threads = grid.num_threads();

    for (VIndex_t vid = start_vid + global_tid; vid < end_vid;
         vid += num_threads) {
        VIndex_t uid = vid - start_vid;

        VIndex_t *neighbors = context.graph_backend.get_neigh(vid);
        VIndex_t neighbors_cnt = context.graph_backend.get_neigh_cnt(vid);
        p_storage.vertex_set[uid].use_copy(neighbors, neighbors_cnt);
        v_storage.subtraction_set[uid].set<0>(vid);
        v_storage.prev_uid[uid * MAX_VERTEXES + 0] = uid;
    }
}

// 找到在 v_storages[curr_pattern_vid][uid] 在 father_pattern_id 深度的 father
// 可以每个 thread（而非 Warp） 找一个不同的 uid
// 边界情况：father_pattern_vid == cur_pattern_vid? 返回 uid
// 边界情况：father_pattern_vid == 0？不能越界
template <Config config>
__device__ inline int find_father(int father_pattern_vid,
                                  const VertexStorage<config> &v_storage,
                                  int uid) {
    return v_storage.prev_uid[uid * MAX_VERTEXES + father_pattern_vid];
}

// 我们希望知道某个 prefix id 是属于哪个 vertex，其实就是最后一个vertex 是什么
template <Config config>
__device__ inline int from_prefix_id_to_vertex_id(
    const DeviceContext<config> &context, int prefix_id) {
    const auto &prefix = context.schedule_data.prefixs[prefix_id];
    return prefix.data[prefix.depth - 1];
}

template <Config config>
__device__ inline int find_prefix_level_uid(
    const DeviceContext<config> &context,
    const VertexStorage<config> &v_storage, int prefix_id, int uid) {
    int vertex_id = from_prefix_id_to_vertex_id(context, prefix_id);
    return find_father(vertex_id, v_storage, uid);
}

// 找到最后一个小于等于这个数的位置
__device__ inline VIndex_t find_prev_index(VIndex_t *sum, int num_units,
                                           VIndex_t target_value) {
    if (num_units == 0) return 0;
    // 获取 nb 最高位的二进制位数
    const VIndex_t p = 32 - __clz(num_units - 1);
    VIndex_t n = 0;
// 每次决定一个二进制位，从高到低
#pragma unroll
    for (int i = p - 1; i >= 0; i--) {
        // 这次决定的是从高往低的第 i 位
        const VIndex_t index = n | (1 << i);
        // 往右侧走
        if (index < num_units && sum[index - 1] <= target_value) {
            n = index;
        }
    }
    return n;
}

template <Config config, int cur_pattern_vid>
__global__ void extend_v_storage_kernel(const DeviceContext<config> context,
                                        PrefixStorage<config> loop_set_storage,
                                        int loop_set_prefix_id,
                                        VertexStorage<config> cur_v_storage,
                                        VertexStorage<config> next_v_storage,
                                        int base_extend_unit_id,
                                        int num_extend_units) {
    using VertexSet = VertexSetTypeDispatcher<config>::type;

    auto grid = cg::this_grid();
    const int global_tid = grid.thread_rank();
    const int num_threads = grid.num_threads();

    for (int base = 0; base < num_extend_units; base += num_threads) {
        int next_extend_uid = base + global_tid;
        if (next_extend_uid >= num_extend_units) continue;

        int uid_in_total = base_extend_unit_id + next_extend_uid;

        // 从下一个点的 uid 反推到上一层的 Unit 和 Vertex，二分查找
        int cur_level_uid =
            find_prev_index(cur_v_storage.unit_extend_sum,
                            cur_v_storage.num_units, uid_in_total);

        for (int i = 0; i <= cur_pattern_vid; i++) {
            next_v_storage.prev_uid[next_extend_uid * MAX_VERTEXES + i] =
                cur_v_storage.prev_uid[cur_level_uid * MAX_VERTEXES + i];
        }
        next_v_storage
            .prev_uid[next_extend_uid * MAX_VERTEXES + cur_pattern_vid + 1] =
            next_extend_uid;

        // 找到 Vertex 在 set 中的 Index
        int base_index = cur_level_uid == 0
                             ? 0
                             : cur_v_storage.unit_extend_sum[cur_level_uid - 1];
        int vertex_index = uid_in_total - base_index;

        // 找到 next_uid 在 loop_set_vertex_id 层的 uid
        int loop_set_uid = find_prefix_level_uid<config>(
            context, next_v_storage, loop_set_prefix_id, next_extend_uid);

        // vertex set + index 获取 vertex_id
        VIndex_t v =
            loop_set_storage.vertex_set[loop_set_uid].get(vertex_index);

        const auto &cur_subtraction_set =
            cur_v_storage.subtraction_set[cur_level_uid];

        next_v_storage.last_level_uid[next_extend_uid] = cur_level_uid;
        next_v_storage.loop_set_uid[next_extend_uid] = loop_set_uid;
        next_v_storage.last_level_v_choose[next_extend_uid] = v;

        // 构造 subtraction_set
        next_v_storage.subtraction_set[next_extend_uid].copy_single_thread(
            cur_subtraction_set);
        next_v_storage.subtraction_set[next_extend_uid]
            .set<cur_pattern_vid + 1>(v);
    }
}

// 设置 v_storage[cur_pattern_vid] 的 prev_uid, subtraction_set
// 设置 p_storage[(cur_pattern_vid + 1) --> prefix_id] 的 vertex_set
template <Config config, int cur_pattern_vid>
__global__ void __maxnreg__(config.engine_config.launch_config.max_regs)
    extend_p_storage_kernel(const DeviceContext<config> context,
                            PrefixStorages<config> p_storages,
                            VertexStorage<config> cur_v_storage,
                            VertexStorage<config> next_v_storage,
                            int num_extend_units) {
    using VertexSet = VertexSetTypeDispatcher<config>::type;
    constexpr auto launch_config = config.engine_config.launch_config;
    constexpr int warps_per_block =
        launch_config.threads_per_block / launch_config.threads_per_warp;
    __shared__ VertexSet new_vertex_sets[warps_per_block];

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<launch_config.threads_per_warp>(block);

    // 下一个点的 loop_set_prefix_id
    const int loop_set_prefix_id =
        context.schedule_data.loop_set_prefix_id[cur_pattern_vid + 1];
    const auto &loop_set_storage = p_storages[loop_set_prefix_id];

    // 下一个点的 prefix_id 的范围
    const int start_prefix_id =
        context.schedule_data.vertex_prefix_start[cur_pattern_vid + 1];
    const int end_prefix_id =
        context.schedule_data.vertex_prefix_start[cur_pattern_vid + 2];

    // load 我们可能需要的所有 Prefix 的信息到 shared memory
    // 这些后面都是公用的
    __shared__ VertexSet *shared_vertex_set[MAX_PREFIXS],
        *father_vertex_set_shared[MAX_PREFIXS];

    const int tid = block.thread_rank();
    if (tid < end_prefix_id - start_prefix_id) {
        shared_vertex_set[tid] = p_storages[start_prefix_id + tid].vertex_set;
        int father_prefix_id =
            context.schedule_data.prefixs_father[start_prefix_id + tid];
        father_vertex_set_shared[tid] =
            father_prefix_id == -1 ? nullptr
                                   : p_storages[father_prefix_id].vertex_set;
    }

    __threadfence_block();
    block.sync();

    const int warp_id = warp.meta_group_rank();
    const int global_wid =
        block.group_index().x * warp.meta_group_size() + warp_id;
    const int num_total_warps = grid.dim_blocks().x * warp.meta_group_size();

    for (int base = 0; base < num_extend_units; base += num_total_warps) {
        int next_extend_uid = base + global_wid;
        if (next_extend_uid >= num_extend_units) continue;

        // 找到 Vertex 在 set 中的 Index
        int cur_level_uid = next_v_storage.last_level_uid[next_extend_uid];
        int loop_set_uid = next_v_storage.loop_set_uid[next_extend_uid];
        VIndex_t v = next_v_storage.last_level_v_choose[next_extend_uid];

        const auto &cur_subtraction_set =
            cur_v_storage.subtraction_set[cur_level_uid];

        // 构建邻居Vertex Set
        // 这两步应该提走，单独封装出去
        VIndex_t *neighbors = context.graph_backend.get_neigh(v);
        VIndex_t neighbors_cnt = context.graph_backend.get_neigh_cnt(v);
        VertexSet &new_vertex_set = new_vertex_sets[warp_id];
        new_vertex_set.init(warp, neighbors, neighbors_cnt);

        bool ap = cur_subtraction_set.has_data<cur_pattern_vid + 1>(warp, v);
        // 构建所有 prefix 对应的 p_storage
        for (int cur_prefix_id = start_prefix_id; cur_prefix_id < end_prefix_id;
             cur_prefix_id++) {
            int index = cur_prefix_id - start_prefix_id;

            bool only_need_size =
                context.schedule_data.prefixs_size_only[cur_prefix_id];

            // 找到下一层的 Storage Unit
            auto &next_vertex_set = shared_vertex_set[index][next_extend_uid];

            if (ap) {
                next_vertex_set.clear();
                continue;
            }

            // 在 GraphSet 这里叫做 build_vertex_set
            // 找到 father prefix id
            int father_prefix_id =
                context.schedule_data.prefixs_father[cur_prefix_id];
            if (father_prefix_id == -1) {
                // 没有 father，直接使用邻居
                next_vertex_set.use_copy(warp, neighbors, neighbors_cnt);
            } else {
                // 如果有 father，那么需要和 father 的 vertex set 取交集
                int father_unit_id = find_prefix_level_uid<config>(
                    context, next_v_storage, father_prefix_id, next_extend_uid);
                const auto &father_vertex_set =
                    father_vertex_set_shared[index][father_unit_id];
                if (!only_need_size) {
                    next_vertex_set.intersect(warp, new_vertex_set,
                                              father_vertex_set);
                } else {
                    // 如果只需要 size，我们就做一个非常简单的相交。
                    // 但这个时候我们需要把 cur_subtraction_set 直接给挖掉
                    next_vertex_set.intersect_size<cur_pattern_vid + 1>(
                        warp, new_vertex_set, father_vertex_set,
                        cur_subtraction_set);
                }
            }
        }
    }
}

// 这个函数构建 cur_pattern_vid 位置的 unit_extend_size
// p_storage 是对应 loop_set_prefix_id 位置的 Prefix
template <Config config, int cur_pattern_vid>
__global__ void prepare_v_storage_kernel(const DeviceContext<config> context,
                                         PrefixStorage<config> p_storage,
                                         VertexStorage<config> v_storage,
                                         int loop_set_prefix_id) {
    const int num_units = v_storage.num_units;
    assert(num_units != 0);

    const int bid = blockIdx.x, tid = threadIdx.x;
    const int global_tid = bid * blockDim.x + tid;
    const int num_threads = gridDim.x * blockDim.x;

    int restrict_index_start =
        context.schedule_data.restrictions_start[cur_pattern_vid + 1];
    int restrict_index_end =
        context.schedule_data.restrictions_start[cur_pattern_vid + 2];

    // 对于每一个本层的 Unit
    // 构建 loop_set_unit 的 size
    for (int base = 0; base < num_units; base += num_threads) {
        int uid = base + global_tid;
        if (uid >= num_units) continue;
        // 找到本层的 unit 在 loop_set_vertex_id 层的 uid
        int loop_set_uid = find_prefix_level_uid<config>(
            context, v_storage, loop_set_prefix_id, uid);
        // printf("uid: %d loop set uid: %d\n", uid, loop_set_uid);
        // 将 uid 位置的 set 写入 unit_extend_size

        // 在这里第一次 apply 限制
        // 不是写 size，而是写比 min_vertex 小的点的个数。
        // 遍历所有和 cur_pattern_vid 相关的限制。
        VIndex_t min_vertex = context.graph_backend.v_cnt();
        for (int i = restrict_index_start; i < restrict_index_end; i++) {
            int restrict_target = context.schedule_data.restrictions[i];
            int v_target = v_storage.subtraction_set[uid].get(restrict_target);
            if (min_vertex > v_target) {
                min_vertex = v_target;
            }
        }
        // 在 loop_set 里面，二分找到第一个大于 min_vertex 的节点

        int size_after_restrict =
            lower_bound(p_storage.vertex_set[loop_set_uid].data(),
                        p_storage.vertex_set[loop_set_uid].size(), min_vertex);
        v_storage.unit_extend_size[uid] = size_after_restrict;
        // v_storage.unit_extend_size[uid] =
        //     p_storage.vertex_set[loop_set_uid].size();
    }
}

constexpr int UID = 2;

// 按照 IEP Info 的提示去计算出每一个 Unit 对应的答案，放到某个数组里面
template <Config config, int cur_pattern_vid>
__global__ void get_iep_answer_kernel(DeviceContext<config> context,
                                      PrefixStorages<config> p_storages,
                                      VertexStorage<config> last_v_storage,
                                      long long *d_ans) {
    // per-thread 去处理 Unit
    const int thread_id = threadIdx.x, block_id = blockIdx.x;
    const int global_tid = block_id * blockDim.x + thread_id;
    const int num_threads = gridDim.x * blockDim.x;

    const int num_units = last_v_storage.num_units;

    const auto &iep_data = context.schedule_data.iep_data;

    int iep_prefix_num = iep_data.iep_prefix_num,
        subgroups_num = iep_data.subgroups_num;

    long long ans[MAX_PREFIXS];

    for (int base = 0; base < num_units; base += num_threads) {
        int uid = base + global_tid;
        if (uid >= num_units) continue;
        long long local_ans = 0;
        // if (uid == UID) {
        //     // subtraction_set
        //     printf("uid: %d subtraction_set: ", uid);
        //     for (int i = 0; i < cur_pattern_vid + 1; i++) {
        //         printf("%d ", last_v_storage.subtraction_set[uid].get(i));
        //     }
        //     printf("\n");
        // }
        for (int prefix_id_x = 0; prefix_id_x < iep_prefix_num; prefix_id_x++) {
            int this_prefix_id = iep_data.iep_vertex_id[prefix_id_x];
            // 需要获得 prefix_id 的 uid
            int father_uid = find_prefix_level_uid<config>(
                context, last_v_storage, this_prefix_id, uid);
            auto &vs = p_storages[this_prefix_id].vertex_set[father_uid];

            if (context.schedule_data.prefixs_size_only[this_prefix_id]) {
                ans[prefix_id_x] = vs.size();
            } else {
                ans[prefix_id_x] =
                    vs.subtraction_size_onethread<cur_pattern_vid + 1>(
                        last_v_storage.subtraction_set[uid]);
            }

            // if (uid == UID) {
            //     // subtraction_set
            //     printf("uid: %d prefix_id: %d father_uid: %d size: %d\n",
            //     uid,
            //            this_prefix_id, father_uid, vs.size());
            //     for (int i = 0; i < vs.size(); i++) {
            //         printf("%d ", vs.get(i));
            //     }
            //     printf("\n");
            // }
        }
        // for (int i = 0; i < iep_prefix_num; i++) {
        //     printf("uid: %d prefix_id: %d ans: %lld size_only: %d\n",
        //     uid, i,
        //            ans[i],
        //            context.schedule_data
        //                .prefixs_size_only[iep_data.iep_vertex_id[i]]);
        // }

        long long val = 1;
        int last_gid = -1;
        for (int gid = 0; gid < subgroups_num; gid++) {
            // if (uid == UID) {
            //     printf("gid: %d ans: %lld\n", gid,
            //            ans[iep_data.iep_ans_pos[gid]]);
            // }
            if (gid == last_gid + 1) {
                val = ans[iep_data.iep_ans_pos[gid]];
            } else {
                val *= ans[iep_data.iep_ans_pos[gid]];
            }
            if (iep_data.iep_flag[gid]) {
                last_gid = gid;
                local_ans += val * iep_data.iep_coef[gid];
                // if (uid == UID) {
                //     printf("uid: %d gid: %d val: %lld coef: %d\n", uid, gid,
                //            val, iep_data.iep_coef[gid]);
                // }
            }
        }
        // printf("uid: %d local_ans: %lld\n", uid, local_ans);
        d_ans[uid] += local_ans;
    }
}

}  // namespace Engine
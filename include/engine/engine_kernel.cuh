#pragma once

#include "engine/context.cuh"
#include "engine/storage.cuh"

namespace Engine {

// 这个 Kernel 是一个一个遍历所有的点，然后构建第一层的 Storage
// 是否可以把这个地方的 遍历 Vertex Set 抽象成遍历一个
// Function，然后在第一层这个 function 就是 identity
// 在其他层可能就是一个很复杂的 function
// 这个 function 是独立的

// space 是单独开的，需要再分给 vertex set 里面的 pointer
template <Config config>
__global__ void set_vertex_set_space_kernel(PrefixStorage<config> p_storage) {
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
            v_storage.prev_uid[uid * MAX_VERTEXES + 0] = uid;
        }
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
    const auto &prefix = context.schedule_data.prefixes[prefix_id];
    return prefix.data[prefix.depth - 1];
}

template <Config config>
__device__ inline int find_prefix_level_uid(
    const DeviceContext<config> &context,
    const VertexStorage<config> &v_storage, int prefix_id, int uid) {
    int vertex_id = from_prefix_id_to_vertex_id(context, prefix_id);
    return find_father(vertex_id, v_storage, uid);
}

__device__ VIndex_t find_prev_index(VIndex_t *sum, int l, int r,
                                    VIndex_t target_value) {
    int mid = 0;
    VIndex_t mid_value = 0;
    while (l != r) {
        mid = (l + r) >> 1;
        mid_value = sum[mid];
        if (mid_value <= target_value) {
            l = mid + 1;
        } else {  // mid_value > target_value
            r = mid;
        }
    }
    return l;
}

template <Config config, int cur_pattern_vid>
__global__ void extend_v_storage_kernel(const DeviceContext<config> context,
                                        PrefixStorage<config> loop_set_storage,
                                        int loop_set_prefix_id,
                                        VertexStorage<config> cur_v_storage,
                                        VertexStorage<config> next_v_storage,
                                        int start_uid, int end_uid) {
    using VertexSet = VertexSetTypeDispatcher<config>::type;
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = gridDim.x * blockDim.x;

    int start_extend_unit_id =
        start_uid == 0 ? 0 : cur_v_storage.unit_extend_sum[start_uid - 1];
    int end_extend_unit_id = cur_v_storage.unit_extend_sum[end_uid - 1];
    int num_extend_units = end_extend_unit_id - start_extend_unit_id;

    for (int base = 0; base < num_extend_units; base += num_threads) {
        int next_extend_uid = base + global_tid;
        if (next_extend_uid >= num_extend_units) continue;

        int uid_in_total = start_extend_unit_id + next_extend_uid;

        // 从下一个点的 uid 反推到上一层的 Unit 和 Vertex，二分查找
        int cur_level_uid =
            find_prev_index(cur_v_storage.unit_extend_sum, start_uid,
                            end_uid - 1, uid_in_total);

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

        // 构造 subtraction_set
        next_v_storage.subtraction_set[next_extend_uid].copy_single_thread(
            cur_subtraction_set);
        next_v_storage.subtraction_set[next_extend_uid]
            .set<cur_pattern_vid + 1>(v);
    }
}

template <Config config, int cur_pattern_vid>
void extend_v_storage(const DeviceContext<config> context,
                      PrefixStorages<config> p_storages,
                      VertexStorages<config> v_storages, int start_uid,
                      int end_uid) {
    const auto &cur_v_storage = v_storages[cur_pattern_vid];
    const auto &next_v_storage = v_storages[cur_pattern_vid + 1];

    // 下一个点的 loop_set_prefix_id
    const int loop_set_prefix_id =
        context.schedule_data.loop_set_prefix_id[cur_pattern_vid + 1];
    const auto &loop_set_storage = p_storages[loop_set_prefix_id];

    extend_v_storage_kernel<config, cur_pattern_vid>
        <<<num_blocks, THREADS_PER_BLOCK>>>(context, loop_set_storage,
                                            loop_set_prefix_id, cur_v_storage,
                                            next_v_storage, start_uid, end_uid);
}

// 设置 v_storage[cur_pattern_vid] 的 prev_uid, subtraction_set
// 设置 p_storage[(cur_pattern_vid + 1) --> prefix_id] 的 vertex_set
template <Config config, int cur_pattern_vid>
__global__ void extend_p_storage_kernel(const DeviceContext<config> context,
                                        PrefixStorages<config> p_storages,
                                        VertexStorages<config> v_storages,
                                        int start_uid, int end_uid) {
    using VertexSet = VertexSetTypeDispatcher<config>::type;
    __shared__ VertexSet new_vertex_sets[WARPS_PER_BLOCK];

    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int global_wid = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // 下一个点的 loop_set_prefix_id
    const int loop_set_prefix_id =
        context.schedule_data.loop_set_prefix_id[cur_pattern_vid + 1];
    const auto &loop_set_storage = p_storages[loop_set_prefix_id];

    // 当前点的 v_storage 和 下一个点的 v_storage
    const auto &cur_v_storage = v_storages[cur_pattern_vid];
    const auto &next_v_storage = v_storages[cur_pattern_vid + 1];

    // 下一个点的 prefix_id 的范围
    const int start_prefix_id =
        context.schedule_data.vertex_prefix_start[cur_pattern_vid + 1];
    const int end_prefix_id =
        context.schedule_data.vertex_prefix_start[cur_pattern_vid + 2];

    // load 我们可能需要的所有 Prefix 到 shared memory
    __shared__ VertexSet *shared_vertex_set[MAX_PREFIXS],
        *father_vertex_set_shared[MAX_PREFIXS];
    const int tid = threadIdx.x;
    if (tid < end_prefix_id - start_prefix_id) {
        shared_vertex_set[tid] = p_storages[start_prefix_id + tid].vertex_set;
        int father_prefix_id =
            context.schedule_data.prefix_fathers[start_prefix_id + tid];
        father_vertex_set_shared[tid] =
            father_prefix_id == -1 ? nullptr
                                   : p_storages[father_prefix_id].vertex_set;
    }

    __threadfence_block();

    // 下一个点的 uid 的数量，开始和结束的位置
    int start_extend_unit_id =
        start_uid == 0 ? 0 : cur_v_storage.unit_extend_sum[start_uid - 1];
    int end_extend_unit_id = cur_v_storage.unit_extend_sum[end_uid - 1];
    int num_extend_units = end_extend_unit_id - start_extend_unit_id;

    for (int base = 0; base < num_extend_units; base += num_total_warps) {
        int next_extend_uid = base + global_wid;
        if (next_extend_uid >= num_extend_units) continue;

        int uid_in_total = start_extend_unit_id + next_extend_uid;

        int cur_level_uid =
            next_v_storage
                .prev_uid[next_extend_uid * MAX_VERTEXES + cur_pattern_vid];

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

        // 构建邻居Vertex Set
        // 这两步应该提走，单独封装出去
        VIndex_t *neighbors = context.graph_backend.get_neigh(v);
        VIndex_t neighbors_cnt = context.graph_backend.get_neigh_cnt(v);
        VertexSet &new_vertex_set = new_vertex_sets[warp_id];
        new_vertex_set.init(neighbors, neighbors_cnt);

        bool appeared = cur_subtraction_set.has_data<cur_pattern_vid + 1>(v);
        // 构建所有 prefix 对应的 p_storage
#pragma unroll
        for (int cur_prefix_id = start_prefix_id; cur_prefix_id < end_prefix_id;
             cur_prefix_id++) {
            int index = cur_prefix_id - start_prefix_id;

            // 找到下一层的 Storage Unit
            auto &next_vertex_set = shared_vertex_set[index][next_extend_uid];

            if (appeared) {
                next_vertex_set.clear();
                continue;
            }

            // 找到 father prefix id
            int father_prefix_id =
                context.schedule_data.prefix_fathers[cur_prefix_id];
            if (father_prefix_id == -1) {
                // 没有 father，直接 copy 邻居
                next_vertex_set.init_copy(neighbors, neighbors_cnt);
            } else {
                // 如果有 father，那么需要和 father 的 vertex set 取交集
                int father_unit_id = find_prefix_level_uid<config>(
                    context, next_v_storage, father_prefix_id, next_extend_uid);
                const auto &father_vertex_set =
                    father_vertex_set_shared[index][father_unit_id];
                next_vertex_set.intersect(new_vertex_set, father_vertex_set);
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
        v_storage.unit_extend_size[uid] =
            p_storage.vertex_set[loop_set_uid].size();
    }
}

template <Config config, int cur_pattern_vid>
void prepare_v_storage(DeviceContext<config> &context,
                       PrefixStorages<config> &prefix_storages,
                       VertexStorages<config> &vertex_storages) {
    const auto &v_storage = vertex_storages[cur_pattern_vid];

    int loop_set_prefix_id =
        context.schedule_data.loop_set_prefix_id[cur_pattern_vid + 1];
    const auto &p_storage = prefix_storages[loop_set_prefix_id];

    prepare_v_storage_kernel<config, cur_pattern_vid>
        <<<num_blocks, THREADS_PER_BLOCK>>>(context, p_storage, v_storage,
                                            loop_set_prefix_id);
}

template <Config config>
__global__ void get_next_unit_kernel(int current_unit, int *next_unit,
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

// 按照 IEP Info 的提示去计算出每一个 Unit 对应的答案，放到某个数组里面
template <Config config, int cur_pattern_vid>
__global__ void get_iep_answer_kernel(DeviceContext<config> context,
                                      PrefixStorages<config> p_storages,
                                      VertexStorages<config> v_storages,
                                      unsigned long long *d_ans) {
    // per-thread 去处理 Unit
    const int thread_id = threadIdx.x, block_id = blockIdx.x;
    const int global_tid = block_id * blockDim.x + thread_id;
    const int num_threads = gridDim.x * blockDim.x;

    const auto &last_v_storage = v_storages[cur_pattern_vid];

    const int num_units = last_v_storage.num_units;

    const auto &iep_data = context.schedule_data.iep_data;

    int iep_prefix_num = iep_data.iep_prefix_num,
        subgroups_num = iep_data.subgroups_num;

    unsigned long long ans[MAX_PREFIXS];

    for (int base = 0; base < num_units; base += num_threads) {
        int uid = base + global_tid;
        if (uid >= num_units) continue;
        unsigned long long local_ans = 0;
        for (int prefix_id_x = 0; prefix_id_x < iep_prefix_num; prefix_id_x++) {
            int this_prefix_id = iep_data.iep_vertex_id[prefix_id_x];
            // 需要获得 prefix_id 的 uid
            int father_uid = find_prefix_level_uid<config>(
                context, last_v_storage, this_prefix_id, uid);
            ans[prefix_id_x] =
                p_storages[this_prefix_id]
                    .vertex_set[father_uid]
                    .subtraction_size_single_thread<cur_pattern_vid>(
                        last_v_storage.subtraction_set[uid]);
        }

        unsigned long long val = 1;
        int last_gid = -1;
        for (int gid = 0; gid < subgroups_num; gid++) {
            if (gid == last_gid + 1) {
                val = ans[iep_data.iep_ans_pos[gid]];
            } else {
                val *= ans[iep_data.iep_ans_pos[gid]];
            }
            if (iep_data.iep_flag[gid]) {
                last_gid = gid;
                local_ans += val * iep_data.iep_coef[gid];
            }
        }
        d_ans[uid] += local_ans;
    }
}
}  // namespace Engine
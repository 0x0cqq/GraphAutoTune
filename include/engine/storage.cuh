#pragma once
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <atomic>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "core/types.hpp"
#include "core/unordered_vertex_set.cuh"
#include "core/vertex_set.cuh"
#include "utils/cuda_utils.cuh"

namespace Engine {

template <Config config>
class StorageUnit {
  public:
    using VertexSet = VertexSetTypeDispatcher<config>::type;
    static_assert(Core::IsVertexSetImpl<VertexSet>);

    // Unordered Vertex Set，记录了目前已经选择的节点的编号
    Core::UnorderedVertexSet<MAX_DEPTH> subtraction_set;
    // Ordered Vertex Set，记录了当前的 Prefix 的 dependency set 相交的结果
    VertexSet vertex_set;

    // 所有的在上面层的 Prefix 在哪里，每层一个
    // 获取 father prefix 的 vertex set: fathers[father_prefix]->vertex_set
    // 然后做 Intersect
    // TODO: 考虑只记录父亲的位置。但是区别不大。
    StorageUnit<config>* fathers[MAX_DEPTH];
};

/**
// 第 LEVEL 层的存储。这个结构体应该仅使用指针。
// 我们会直接在 GPU 上。之后可以考虑多架构。
// 默认情况下，这个结构体的所有空间都是 0
template <Config config>
class LevelStorage {
  private:
    // 每一个 BLOCK_SIZE 个 VIndex_t 的存储块。每个 VertexSet 都是按照
    // BLOCK_SIZE 的倍数分配的。
    static constexpr int BLOCK_SIZE = 4000;
    static constexpr int NUM_BLOCKS = 1000;
    // 该层总共的内存大小。
    static constexpr int TOTAL_SIZE = BLOCK_SIZE * NUM_BLOCKS;
    // 本层的存储空间
    VIndex_t _storage[TOTAL_SIZE];
    // 本层的 StorageUnit 指针
    StorageUnit<config> _storage_unit[NUM_BLOCKS];
    // 已经分配出去的块数
    int _allocated_blocks;
    // 已经分配出去的 StorageUnit 数量
    int _allocated_storage_units;

    // 本层已经 extend 的 vertex 位置
    int _cur_storage_unit;  // 当前在处理 Vertex Set 的 index
    int _cur_vertex_index;  // 当前在处理 vertex 在 Set 中的 index

  public:
    __device__ bool extend_finished() const {
        return _cur_storage_unit == _allocated_storage_units - 1 &&
               _cur_vertex_index ==
                   _storage_unit[_cur_storage_unit].vertex_set.size();
    }

    __device__ int cur_storage_unit() const { return _cur_storage_unit; }

    __device__ int cur_vertex_index() const { return _cur_vertex_index; }

    // 形式上清空该层的存储
    __device__ void clear() {
        _allocated_blocks = 0;
        _allocated_storage_units = 0;
        _cur_storage_unit = 0;
        _cur_vertex_index = 0;
    }

    // 注意：每个 Worker 只有一个线程调用这个函数
    __device__ VIndex_t* allocate(VIndex_t max_vertex_set_size) {
        int blocks_needed = (max_vertex_set_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // 如果剩余的块数小于总块数，那么就返回
        // nullptr，告诉调用者无法分配。 Note: 这里需要
        // atomic，因为可能有多个线程同时调用这个函数。
        int last_allocated_blocks =
            atomicAdd(&_allocated_blocks, blocks_needed);
        if (last_allocated_blocks + blocks_needed > NUM_BLOCKS) {
            // 放回去，或许有其他线程还可以塞一些东西进去。
            atomicSub(&_allocated_blocks, blocks_needed);
            return nullptr;
        } else {
            // 多分配了一个 vertex set
            atomicAdd(&_allocated_storage_units, 1);
            // 返回分配的内存块的指针。
            return _storage + last_allocated_blocks * BLOCK_SIZE;
        }
    }
};

*/

template <Config config>
class LevelStorage {
  private:
    static constexpr int MAX_SET_SIZE = 2000;  // 每个 Set 最多 2000 个数
    static constexpr int NUMS_STORAGE_UNIT = 1000;        // 1000 个 Vertex Set
    VIndex_t _storage[MAX_SET_SIZE * NUMS_STORAGE_UNIT];  // 存储空间
    StorageUnit<config> _storage_unit[NUMS_STORAGE_UNIT];
    int _cur_storage_unit;  // 当前处理到的 Vertex Set 的 index
    int _cur_vertex_index;  // 当前处理到的 vertex 在 Set 中的 index

    int _allocated_storage_units;  // 已经分配出去的 Storage Unit 的个数

  public:
    __device__ const StorageUnit<config>& storage_unit(int i) const {
        return _storage_unit[i];
    }

    __device__ StorageUnit<config>& storage_unit(int i) {
        return _storage_unit[i];
    }

    __device__ bool extend_finished() const {
        return _cur_storage_unit == _allocated_storage_units - 1 &&
               _cur_vertex_index ==
                   _storage_unit[_cur_storage_unit].vertex_set.size();
    }

    __device__ int cur_storage_unit() const { return _cur_storage_unit; }

    __device__ int cur_vertex_index() const { return _cur_vertex_index; }

    __device__ int allocated_storage_units() const {
        return _allocated_storage_units;
    }

    __device__ void clear() {
        _cur_storage_unit = 0;
        _cur_vertex_index = 0;
        _allocated_storage_units = 0;
    }
    __device__ VIndex_t* allocate() {
        int last_allocated_units = atomicAdd(&_allocated_storage_units, 1);
        if (last_allocated_units > NUMS_STORAGE_UNIT) {
            atomicSub(&_allocated_storage_units, 1);
            return nullptr;
        } else {
            return _storage + last_allocated_units * MAX_SET_SIZE;
        }
    }
};
}  // namespace Engine

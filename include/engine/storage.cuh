#pragma once
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <atomic>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "core/types.hpp"
#include "core/unordered_vertex_set.cuh"
#include "core/vertex_set.cuh"

namespace Engine {

// 包括 Ordered Vertex Set 和 Unordered Vertex Set 的信息，有一个指针连往外侧
template <Config config>
class StorageUnit {
    Core::UnorderedVertexSet<MAX_DEPTH> subtraction_set;

    if constexpr (true) {
        using Impl = Config;
    } else {
        
    }
};

// 第 LEVEL 层的存储。
template <Config config>
class LevelStorage {
    // 每一个 BLOCK_SIZE 个 VIndex_t 的存储块。每个 VertexSet 都是按照
    // BLOCK_SIZE 的倍数分配的。
    static constexpr int BLOCK_SIZE = 4000;
    static constexpr int NUM_BLOCKS = 1000;
    // 该层总共的内存大小。
    static constexpr int TOTAL_SIZE = BLOCK_SIZE * NUM_BLOCKS;
    // level
    int _level;
    // 本层的存储指针
    VIndex_t* _storage;
    // 本层的 StorageUnit 指针
    StorageUnit<Impl>* _storage_unit;
    // 已经分配出去的块数
    int _allocated_blocks;
    // 已经分配出去的 StorageUnit 数量
    int _allocated_storage_units;

    __device__ void init(int level, VIndex_t* storage,
                         StorageUnit<Impl>* storage_unit) {
        _storage = storage;
        _storage_unit = storage_unit;
        _allocated_blocks = 0;
        _allocated_storage_units = 0;
        _level = level;
    }

    __device__ void clear() {
        _allocated_blocks = 0;
        _allocated_storage_units = 0;
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
}  // namespace Engine

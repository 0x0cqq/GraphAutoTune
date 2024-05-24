#pragma once
#include <thrust/reduce.h>

#include <atomic>
#include <nvfunctional>

#include "configs/config.hpp"
#include "consts/general_consts.hpp"
#include "core/types.hpp"
#include "core/unordered_vertex_set.cuh"
#include "core/vertex_set.cuh"
#include "utils/cuda_utils.cuh"

// storage 只有 指针！

namespace Engine {

template <Config config>
class VertexStorage {
    using VertexSet = VertexSetTypeDispatcher<config>::type;

  public:
    DeviceType _device_type;
    // Unordered Vertex Set，记录了目前已经选择的节点的编号
    Core::UnorderedVertexSet<MAX_VERTEXES>* subtraction_set;
    // 本层的 Unit 在上一个节点的对应的 Index 在哪
    int* prev_uid;
    // 本层总共使用了多少 Unit
    VIndex_t num_units;
    VIndex_t max_units;

    // 这个 Unit 对应的 Loop Set 有多少点，也就是能伸出多少触角
    VIndex_t* unit_extend_size;
    // 上面的前缀和
    VIndex_t* unit_extend_sum;

    // 上一层对应的信息
    VIndex_t* last_level_uid;
    VIndex_t* last_level_v_choose;

    // Loop Set 的 vertex
    VIndex_t* loop_set_uid;

    // void* d_temp_storage;
    // size_t temp_storage_bytes;

    __host__ void init(int max_units_, DeviceType device_type = GPU_DEVICE) {
        _device_type = device_type;
        if (device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaMalloc(
                &subtraction_set,
                sizeof(Core::UnorderedVertexSet<MAX_VERTEXES>) * max_units_));
            gpuErrchk(
                cudaMalloc(&prev_uid, sizeof(int) * max_units_ * MAX_VERTEXES));
            gpuErrchk(
                cudaMalloc(&unit_extend_size, sizeof(VIndex_t) * max_units_));
            gpuErrchk(
                cudaMalloc(&unit_extend_sum, sizeof(VIndex_t) * max_units_));
            gpuErrchk(
                cudaMalloc(&last_level_uid, sizeof(VIndex_t) * max_units_));
            gpuErrchk(cudaMalloc(&last_level_v_choose,
                                 sizeof(VIndex_t) * max_units_));
            gpuErrchk(cudaMalloc(&loop_set_uid, sizeof(VIndex_t) * max_units_));

            num_units = 0;
            max_units = max_units_;
        } else if (device_type == DeviceType::CPU_DEVICE) {
            subtraction_set =
                new Core::UnorderedVertexSet<MAX_VERTEXES>[max_units_];
            prev_uid = new int[max_units_ * MAX_VERTEXES];
            unit_extend_size = new VIndex_t[max_units_];
            unit_extend_sum = new VIndex_t[max_units_];

            last_level_uid = new VIndex_t[max_units_];
            last_level_v_choose = new VIndex_t[max_units_];

            loop_set_uid = new VIndex_t[max_units_];

            num_units = 0;
            max_units = max_units_;
        } else {
            assert(false);
        }
    }

    __host__ void destroy() {
        if (_device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaFree(subtraction_set));
            gpuErrchk(cudaFree(prev_uid));
            gpuErrchk(cudaFree(unit_extend_size));
            gpuErrchk(cudaFree(unit_extend_sum));
            gpuErrchk(cudaFree(last_level_uid));
            gpuErrchk(cudaFree(last_level_v_choose));
            gpuErrchk(cudaFree(loop_set_uid));
        } else if (_device_type == DeviceType::CPU_DEVICE) {
            delete[] subtraction_set;
            delete[] prev_uid;
            delete[] unit_extend_size;
            delete[] unit_extend_sum;
            delete[] last_level_uid;
            delete[] last_level_v_choose;
            delete[] loop_set_uid;
        } else {
            assert(false);
        }
    }
};

template <Config config>
class PrefixStorage {
    using VertexSet = VertexSetTypeDispatcher<config>::type;

  public:
    // 空间的位置 CPU or GPU
    DeviceType _device_type;
    // 存储 Vertex 的空间
    VIndex_t* space;  // LEN = num_units * set_size
    // Ordered Vertex Set，记录了当前的 Prefix 的 dependency set 相交的结果
    VertexSet* vertex_set;  // LEN = num_units

  public:
    // 禁用复制构造函数
    __host__ void init(int num_units, int set_size,
                       DeviceType device_type = GPU_DEVICE) {
        _device_type = device_type;
        if (device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(
                cudaMalloc(&space, sizeof(VIndex_t) * set_size * num_units));
            gpuErrchk(cudaMalloc(&vertex_set, sizeof(VertexSet) * num_units));
        } else if (device_type == DeviceType::CPU_DEVICE) {
            space = new VIndex_t[set_size * num_units];
            vertex_set = new VertexSet[num_units];
        } else {
            assert(false);
        }
    }
    __host__ void destroy() {
        if (_device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaFree(space));
            gpuErrchk(cudaFree(vertex_set));
        } else if (_device_type == DeviceType::CPU_DEVICE) {
            delete[] space;
            delete[] vertex_set;
        } else {
            assert(false);
        }
    }
};

template <Config config>
struct PrefixStorages {
    PrefixStorage<config> storage[MAX_PREFIXS];
    // operator [] 重载
    __host__ __device__ PrefixStorage<config>& operator[](int index) {
        return storage[index];
    }
};

template <Config config>
struct VertexStorages {
    VertexStorage<config> storage[MAX_VERTEXES];
    // operator [] 重载
    __host__ __device__ VertexStorage<config>& operator[](int index) {
        return storage[index];
    }
};

}  // namespace Engine

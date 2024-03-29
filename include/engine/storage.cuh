#pragma once
#include <atomic>
#include <nvfunctional>

#include "configs/config.hpp"
#include "configs/gpu_consts.cuh"
#include "core/types.hpp"
#include "core/unordered_vertex_set.cuh"
#include "core/vertex_set.cuh"
#include "utils/cuda_utils.cuh"

// storage 只有 指针！

namespace Engine {

constexpr int MAX_SET_SIZE = 200;  // 每个 Set 最多 x 个数
constexpr int NUMS_UNIT = 100000;  // y 个 Vertex Set

template <Config config>
class VertexStorage {
  public:
    DeviceType _device_type;
    // Unordered Vertex Set，记录了目前已经选择的节点的编号
    Core::UnorderedVertexSet<MAX_VERTEXES>* subtraction_set;
    // 该 Unit 在上一个节点的对应的 Index 在哪
    int* previous_index;

    // void* d_temp_storage;
    // size_t temp_storage_bytes;

    // 这个 Unit 对应的 Loop Set 有多少点，也就是能伸出多少触角
    VIndex_t* unit_extend_size;
    // 上面的前缀和
    VIndex_t* unit_extend_sum;
    // 使用了多少的 unit
    VIndex_t num_units;

    __host__ void init(DeviceType device_type = GPU_DEVICE) {
        _device_type = device_type;
        if (device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaMalloc(
                &subtraction_set,
                sizeof(Core::UnorderedVertexSet<MAX_VERTEXES>) * NUMS_UNIT));
            gpuErrchk(cudaMalloc(&previous_index, sizeof(int) * NUMS_UNIT));
            gpuErrchk(
                cudaMalloc(&unit_extend_size, sizeof(VIndex_t) * NUMS_UNIT));
            gpuErrchk(
                cudaMalloc(&unit_extend_sum, sizeof(VIndex_t) * NUMS_UNIT));
            num_units = 0;

            // cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
            //                               p_storage.unit_extend_size,
            //                               p_storage.unit_extend_sum,
            //                               NUMS_UNIT);

            // // Allocate temporary storage for inclusive prefix sum
            // gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        } else if (device_type == DeviceType::CPU_DEVICE) {
            subtraction_set =
                new Core::UnorderedVertexSet<MAX_VERTEXES>[NUMS_UNIT];
            previous_index = new int[NUMS_UNIT];
            unit_extend_size = new VIndex_t[NUMS_UNIT];
            unit_extend_sum = new VIndex_t[NUMS_UNIT];

            num_units = 0;
            // d_temp_storage = nullptr;
            // temp_storage_bytes = 0;
        } else {
            assert(false);
        }
    }

    __host__ void destroy() {
        if (_device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaFree(subtraction_set));
            gpuErrchk(cudaFree(previous_index));
            gpuErrchk(cudaFree(unit_extend_size));
            gpuErrchk(cudaFree(unit_extend_sum));
            // gpuErrchk(cudaFree(d_temp_storage));
        } else if (_device_type == DeviceType::CPU_DEVICE) {
            delete[] subtraction_set;
            delete[] previous_index;
            delete[] unit_extend_size;
            delete[] unit_extend_sum;
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
    VIndex_t* space;  // LEN = NUMS_UNIT * MAX_SET_SIZE
    // Ordered Vertex Set，记录了当前的 Prefix 的 dependency set 相交的结果
    VertexSet* vertex_set;  // LEN = NUMS_UNIT

  public:
    // 禁用复制构造函数
    __host__ void init(DeviceType device_type = GPU_DEVICE) {
        _device_type = device_type;
        if (device_type == DeviceType::GPU_DEVICE) {
            gpuErrchk(cudaMalloc(&space,
                                 sizeof(VIndex_t) * MAX_SET_SIZE * NUMS_UNIT));
            gpuErrchk(cudaMalloc(&vertex_set, sizeof(VertexSet) * NUMS_UNIT));
        } else if (device_type == DeviceType::CPU_DEVICE) {
            space = new VIndex_t[MAX_SET_SIZE * NUMS_UNIT];
            vertex_set = new VertexSet[NUMS_UNIT];
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

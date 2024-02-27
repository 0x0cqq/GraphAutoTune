#pragma once
#include <array>

#include "core/types.hpp"

namespace GPU {

template <size_t MAX_SIZE>
class UnorderedVertexSet {
    std::array<VIndex_t, MAX_SIZE> data;

    template <size_t current_pos>
    requires requires(current_pos < MAX_SIZE)
    __device__ __host__ void set(VIndex_t index) {
        data[current_pos] = index;
    }

    template <size_t current_pos>
    requires requires(current_pos < MAX_SIZE)
    __device__ __host__ void remove() {
        // 不需要做什么
    }

    template <size_t current_pos>
    requires requires(current_pos < MAX_SIZE)
    __device__ __host__ bool has_data(VIndex_t value) {
        for (size_t i = 0; i < current_pos; i++) {
            if (data[i] == value) {
                return true;
            }
        }
        return false;
    }
};

}  // namespace GPU
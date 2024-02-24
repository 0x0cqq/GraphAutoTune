#pragma once
#include <atomic>

#include "types.hpp"

class LevelStorage {
    static constexpr int BLOCK_SIZE = 4000;
    VIndex_t* candidates;
    int* father;
    std::atomic<size_t> allocated_size;
};

#pragma once
#include <array>

#include "core/types.hpp"
#include "engine/storage.hpp"
#include "engine/worker.hpp"

class Engine {
    constexpr static int MAX_DEPTH = 10;
    template <int depth>
    void extend(const LevelStorage &last, LevelStorage &current,
                WorkerInfo &worker);

    template <int depth>
    void search(std::array<LevelStorage, MAX_DEPTH> &storages,
                WorkerInfo &worker);
};

#pragma once
#include <algorithm>

#include "core/types.hpp"

class WorkerInfo {
    int worker_id;
    VIndex_t current_vertex;
    std::pair<VIndex_t, VIndex_t> current_progress;
};

#pragma once
#include <algorithm>

#include "core/types.hpp"

namespace Engine {

// 这个是 Per Worker 的信息
class WorkerInfo {
    VIndex_t current_vertex;
    std::pair<VIndex_t, VIndex_t> current_progress;
};

}  // namespace Engine

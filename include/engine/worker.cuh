#pragma once
#include <algorithm>

#include "core/types.hpp"

namespace Engine {

// 这个是 Per Worker 的信息
// 在 GPU 上就会放到
class WorkerInfo {
  public:
    unsigned long long local_answer;
    void clear() {
        local_answer = 0;
    }
};

}  // namespace Engine

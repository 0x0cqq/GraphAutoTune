// 在 GPU 上进行 Pattern Matching.
#include <algorithm>

#include "configs/project_consts.hpp"
#include "core/schedule.hpp"
#include "engine/engine.cuh"
#include "infra/graph_backend.cuh"
#include "utils/cuda_utils.cuh"

constexpr Config default_config{};

int main() {
    // 数据图
    std::ifstream graph_file{PROJECT_ROOT / "data/test_graph.bin"};
    // 模式图
    std::string_view pattern_str{"0100110110010110110010100"};  // house

    // 1. 构建 Context
    // Schedule
    Core::Schedule schedule{pattern_str};
    // 图后端
    Infra::GlobalMemoryGraph<default_config> graph{graph_file, true};

    // 设备上下文
    Engine::DeviceContext<default_config> context{schedule, graph};
    std::cout << "Size of Device Context: " << sizeof(context) << " Bytes"
              << std::endl;

    context.to_device();

    // 2. 构建 Engine
    Engine::Executor<default_config> engine{GPU_DEVICE};
    std::cout << "Size of Executor: " << sizeof(engine) << " Bytes"
              << std::endl;

    // 3. 进行 Match
    std::cout << "Enter the kernel..." << std::endl;

    unsigned long long ans =
        Engine::pattern_matching<default_config>(engine, context);

    // 4. 输出结果

    std::cout << "Answer: " << ans << std::endl;

    // 输出结果
    return 0;
}
// 在 GPU 上进行 Pattern Matching.
#include <algorithm>

#include "configs/project_consts.hpp"
#include "core/schedule.hpp"
#include "engine/engine.cuh"
#include "infra/graph_backend.cuh"
#include "utils/cuda_utils.cuh"

constexpr Config default_config{};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        // too few arguments
        std::cerr << "Too few arguments" << std::endl;
        return 1;
    }
    std::string graph_name{argv[1]};
    std::cout << "Graph Name: " << graph_name << std::endl;
    // 数据图
    std::ifstream graph_file{PROJECT_ROOT / graph_name};
    if (!graph_file.is_open()) {
        std::cerr << "Cannot open the graph file" << std::endl;
        return 1;
    }
    // 模式图
    std::string pattern_str{argv[2]};
    // std::string_view pattern_str{"0111101111011110"};            // 4-clique
    // std::string_view pattern_str{"0100110110010110110010100"};   // house

    // 1. 构建 Context
    // Schedule
    Core::Schedule schedule{pattern_str};
    schedule.output();
    // 图后端
    Infra::GlobalMemoryGraph<default_config> graph{graph_file, true};
    graph.output();

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
    std::cout << "Enter the Search..." << std::endl;

    auto time_start = std::chrono::high_resolution_clock::now();

    unsigned long long ans = engine.perform_search(context);

    auto time_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        time_end - time_start)
                        .count();

    std::cout << "Time: " << duration << " microseconds ("
              << double(duration) / 100000 << " s)" << std::endl;

    // 4. 输出结果

    std::cout << "Answer: " << ans << std::endl;

    // 输出结果
    return 0;
}
#include "configs/project_consts.hpp"
#include "infra/graph_backend.cuh"

constexpr Config config{};

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        std::cerr << "Too less arguments" << std::endl;
        return 1;
    }
    std::string file_name{argv[1]};
    std::ifstream graph_input{PROJECT_ROOT / "data" / (file_name + ".txt")};
    if (!graph_input.is_open()) {
        std::cerr << "Failed to open graph file" << std::endl;
        return 1;
    }

    std::ofstream graph_output{PROJECT_ROOT / "data" / (file_name + ".bin")};
    if (!graph_output.is_open()) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }

    Infra::GlobalMemoryGraph<config> graph{graph_input, false};
    graph.export_to_binary_file(graph_output);
    std::cout << "Graph exported to binary file" << std::endl;
    return 0;
}
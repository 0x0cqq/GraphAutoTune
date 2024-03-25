#include "infra/graph_backend.cuh"

constexpr Config config{};

int main() {
    std::ifstream graph_input = std::ifstream("../data/data_graph_30.txt");
    if (!graph_input.is_open()) {
        std::cerr << "Failed to open graph file" << std::endl;
        return 1;
    }

    std::ofstream graph_output = std::ofstream("../data/data_graph_30.bin");
    if (!graph_output.is_open()) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }

    Infra::GlobalMemoryGraph<config> graph{graph_input, false};
    graph.export_to_binary_file(graph_output);
    std::cout << "Graph exported to binary file" << std::endl;
    return 0;
}
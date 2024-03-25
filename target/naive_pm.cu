#include <fstream>
#include <string>

#include "configs/project_consts.hpp"
#include "core/pattern.hpp"
#include "infra/graph_backend.cuh"

constexpr Config default_config{};

template <Config config>
int pattern_matching_func(int depth, std::vector<int> &choose_vertexes,
                          Infra::GlobalMemoryGraph<config> &graph,
                          Core::Pattern &pattern) {
    if (depth == pattern.v_cnt()) {
        std::cout << "Pattern found: ";
        for (int i = 0; i < choose_vertexes.size(); i++) {
            std::cout << choose_vertexes[i] << " ";
        }
        std::cout << std::endl;
        return 1;
    }
    int ans = 0;
    int n = graph.v_cnt();
    for (int i = 0; i < n; i++) {
        bool flag = true;
        for (int j = 0; j < choose_vertexes.size(); j++) {
            bool exist_in_pattern = pattern.has_edge(j, depth);
            bool exist_in_graph = graph.has_edge(choose_vertexes[j], i);
            if (exist_in_pattern && !exist_in_graph) {
                flag = false;
                break;
            }
        }
        if (flag) {
            choose_vertexes.push_back(i);
            int result = pattern_matching_func(depth + 1, choose_vertexes,
                                               graph, pattern);
            ans += result;
            choose_vertexes.pop_back();
        }
    }
    return ans;
}

template <Config config>
int pattern_matching(Infra::GlobalMemoryGraph<config> &graph,
                     Core::Pattern &pattern) {
    std::vector<int> choose_vertexes;
    return pattern_matching_func(0, choose_vertexes, graph, pattern);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        // too few arguments
        std::cerr << "Too few arguments" << std::endl;
        return 1;
    }
    std::string graph_name{argv[1]};
    std::ifstream graph_bin{PROJECT_ROOT / graph_name};
    if (!graph_bin.is_open()) {
        std::cerr << "Cannot open the graph file" << std::endl;
        return 1;
    }
    std::string_view pattern_str{"0111101111011110"};

    Infra::GlobalMemoryGraph<default_config> data_graph{graph_bin, true};

    Core::Pattern pattern{pattern_str};

    std::cout << pattern_matching(data_graph, pattern) << std::endl;
    return 0;
}
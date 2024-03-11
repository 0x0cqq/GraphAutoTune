#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>

#include "infra/graph_backend.cuh"

constexpr Config default_config{};

std::string project_dir = "/home/cqq/GraphMining/GraphAutoTuner";

TEST_CASE("Graph Test", "[graph]") {
    std::ifstream graph_text{project_dir + std::string{"/data/test_graph.txt"}};
    REQUIRE(graph_text.is_open());

    std::ifstream graph_bin{project_dir + std::string{"/data/test_graph.bin"}};
    REQUIRE(graph_bin.is_open());

    Infra::GlobalMemoryGraph<default_config> graph_1{graph_text, false};
    Infra::GlobalMemoryGraph<default_config> graph_2{graph_bin, true};

    REQUIRE(graph_1.v_cnt() == graph_2.v_cnt());
    REQUIRE(graph_1.e_cnt() == graph_2.e_cnt());

    for (VIndex_t i = 0; i <= graph_1.v_cnt(); i++) {
        REQUIRE(graph_1.vertexes()[i] == graph_2.vertexes()[i]);
    }

    for (EIndex_t i = 0; i < graph_1.e_cnt(); i++) {
        REQUIRE(graph_1.edges()[i] == graph_2.edges()[i]);
    }
}
#include <catch2/catch_test_macros.hpp>

#include "core/schedule.hpp"

TEST_CASE("Schedule Test", "[schedule]") {
    // Core::Pattern p{"0110010101101010010101010"};
    Core::Pattern p{"0100110110010110110010100"};

    Core::Schedule schedule{p};

    schedule.output();
}

TEST_CASE("Graph Cnt Calculate", "[graph cnt]") {
    const auto graph_cnt = Core::calculate_graph_cnt(5);

    // output graph_cnt
    for (int i = 0; i < graph_cnt.size(); i++) {
        std::cout << i << ": " << graph_cnt[i].first << " "
                  << graph_cnt[i].second << std::endl;
    }
}

TEST_CASE("Generate IEP Group Info Test", "[iep group]") {
    const int iep_suffix_vertexes = 5;
    const auto graph_cnt = Core::calculate_graph_cnt(iep_suffix_vertexes);

    std::vector<VIndex_t> id(iep_suffix_vertexes);
    std::vector<Core::IEPGroup> groups;
    get_iep_groups(0, id, 0, iep_suffix_vertexes, groups, graph_cnt);

    // 打印所有的组合
    for (const auto &group : groups) {
        group.output(true);
    }
}
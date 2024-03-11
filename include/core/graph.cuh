#pragma once

#include "configs/config.hpp"
#include "core/types.hpp"
#include "infra/graph_backend.cuh"

template <typename T>
concept isGraphBackendImpl = requires(T t, VIndex_t v) {
    // 图的基本信息
    { t.v_cnt() } -> std::same_as<VIndex_t>;
    { t.e_cnt() } -> std::same_as<EIndex_t>;
    // 获得节点的邻居的信息。
    { t.get_neigh(v) } -> std::same_as<VIndex_t*>;
    { t.get_neigh_cnt(v) } -> std::same_as<VIndex_t>;
};

// graph
template <Config config>
requires(config.infra_config.graph_backend_type == InMemory)
class GraphBackendTypeDispatcher<config> {
  public:
    using type = Infra::GlobalMemoryGraph<config>;
};
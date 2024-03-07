#pragma once
#include <string_view>
#include <vector>

#include "configs/config.hpp"
#include "core/types.hpp"
#include "infra/graph_backend.cuh"

namespace Core {

class Graph {
  public:
    VIndex_t _v_cnt;
    EIndex_t _e_cnt;
};

template <typename T>
concept isGraphBackendImpl = requires(T t, VIndex_t v) {
    // 图的基本信息
    { t.v_cnt() } -> std::same_as<VIndex_t>;
    { t.e_cnt() } -> std::same_as<EIndex_t>;
    // 获得节点的邻居的信息。
    { t.get_neigh(v) } -> std::same_as<VIndex_t*>;
    { t.get_neigh_cnt(v) } -> std::same_as<VIndex_t>;
};

class Pattern {
  private:
    VIndex_t _v_cnt;
    EIndex_t _e_cnt;
    std::vector<int8_t> _edges;

  public:
    VIndex_t v_cnt() const { return _v_cnt; }
    EIndex_t e_cnt() const { return _e_cnt; }
    Pattern(VIndex_t v_cnt)
        : _v_cnt(v_cnt), _e_cnt(0), _edges(v_cnt * v_cnt, 0) {}
    Pattern(const std::string_view& adj_mat)
        : _v_cnt(0), _e_cnt(0), _edges(adj_mat.size(), 0) {
        int len = adj_mat.size();
        for (int n = 1; n < len; n++) {
            if (n * n == len) {
                _v_cnt = n;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < i; j++) {
                        if (adj_mat[i * n + j] == '1' ||
                            adj_mat[j * n + i] == '1') {
                            add_edge(i, j);
                        }
                    }
                }
                break;
            }
        }
    }

    void add_edge(VIndex_t u, VIndex_t v) {
        _edges[u * _v_cnt + v] = 1;
        _edges[v * _v_cnt + u] = 1;
        _e_cnt++;
    }

    bool has_edge(VIndex_t u, VIndex_t v) const {
        return _edges[u * _v_cnt + v] == 1;
    }
};

}  // namespace Core

template <Config config>
requires(config.infra_config.graph_backend_type == InMemory)
class GraphBackendTypeDispatcher<config> {
  public:
    using type = Infra::GlobalMemoryGraph<config>;
};
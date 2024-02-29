#pragma once
#include <vector>

#include "core/types.hpp"

namespace Core {

class Graph {
  public:
    VIndex_t _v_cnt;
    EIndex_t _e_cnt;
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
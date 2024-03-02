#pragma once
#include <algorithm>
#include <concepts>

#include "configs/config.hpp"
#include "core/types.hpp"

namespace Infra {

template <typename T>
concept isGraphBackend = requires(T t, VIndex_t v) {
    // 图的基本信息
    { t.v_cnt() } -> std::same_as<VIndex_t>;
    { t.e_cnt() } -> std::same_as<EIndex_t>;
    // 获得节点的邻居的信息。
    { t.get_neigh(v) } -> std::same_as<VIndex_t *>;
    { t.get_neigh_cnt(v) } -> std::same_as<VIndex_t>;
};

// 和 GraphSet 里面一样的，最简单的 CSR backend
template <Config config>
class GlobalMemoryGraph {
  private:
    VIndex_t _v_cnt;
    EIndex_t _e_cnt;
    EIndex_t *_vertexes;  // The start & edge of the edge set
    VIndex_t *_edges;     // The edge set;
  public:
    __host__ __device__ VIndex_t v_cnt() const { return this->_v_cnt; }
    __host__ __device__ EIndex_t e_cnt() const { return this->_e_cnt; }
    __device__ VIndex_t *get_neigh(VIndex_t v) const {
        return _edges + _vertexes[v];
    }
    __device__ VIndex_t get_neigh_cnt(VIndex_t v) const {
        return _vertexes[v + 1] - _vertexes[v];
    }
};

}  // namespace Infra
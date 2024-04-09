#pragma once

#include <iostream>
#include <string_view>
#include <vector>

#include "core/types.hpp"

namespace Core {

class Pattern {
  private:
    VIndex_t _v_cnt;
    EIndex_t _e_cnt;
    std::vector<int8_t> _edges;

  public:
    constexpr VIndex_t v_cnt() const { return _v_cnt; }
    constexpr EIndex_t e_cnt() const { return _e_cnt; }
    constexpr Pattern(VIndex_t v_cnt)
        : _v_cnt(v_cnt), _e_cnt(0), _edges(v_cnt * v_cnt, 0) {}
    constexpr Pattern(const std::string &adj_mat)
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

    constexpr void add_edge(VIndex_t u, VIndex_t v) {
        _edges[u * _v_cnt + v] = 1;
        _edges[v * _v_cnt + u] = 1;
        _e_cnt++;
    }

    constexpr bool has_edge(VIndex_t u, VIndex_t v) const {
        return _edges[u * _v_cnt + v] == 1;
    }

    constexpr bool operator==(const Pattern &other) const {
        return _v_cnt == other._v_cnt && _edges == other._edges;
    }

    constexpr int get_max_degree() const {
        int max_degree = 0;
        for (VIndex_t i = 0; i < _v_cnt; i++) {
            int degree = 0;
            for (VIndex_t j = 0; j < _v_cnt; j++) {
                if (_edges[i * _v_cnt + j] == 1) {
                    degree++;
                }
            }
            if (degree > max_degree) {
                max_degree = degree;
            }
        }
        return max_degree;
    }
};

// helper functions

constexpr Pattern get_permutated_pattern(
    const std::vector<int> &permutation_order, const Pattern &p) {
    Pattern new_p{p.v_cnt()};
    for (int i = 0; i < p.v_cnt(); i++) {
        for (int j = 0; j < p.v_cnt(); j++) {
            if (p.has_edge(i, j))
                new_p.add_edge(permutation_order[i], permutation_order[j]);
        }
    }
    return new_p;
}

// 每一个点都必须与排在前面的某个节点相连
constexpr bool is_pattern_valid(const Pattern &p) {
    // every point(except the first one) must connect to some one previous in
    // the permutation
    for (int i = 1; i < p.v_cnt(); i++) {
        bool has_edge = false;
        for (int j = 0; j < i; j++) {
            if (p.has_edge(i, j)) {
                has_edge = true;
                break;
            }
        }
        if (!has_edge) {
            return false;
        }
    }
    return true;
};

void output_pattern(const Pattern &p) {
    std::cout << "Pattern:" << std::endl;
    std::cout << "  V: " << p.v_cnt() << " E: " << p.e_cnt() << std::endl;
    std::cout << "  Edges: ";
    for (int i = 0; i < p.v_cnt(); i++) {
        for (int j = i; j < p.v_cnt(); j++) {
            if (p.has_edge(i, j)) {
                std::cout << "(" << i << "," << j << ") ";
            }
        }
    }
    std::cout << std::endl;
}

}  // namespace Core
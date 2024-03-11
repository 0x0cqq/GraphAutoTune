#pragma once
#include <algorithm>
#include <concepts>
#include <fstream>
#include <iostream>
#include <vector>

#include "configs/config.hpp"
#include "core/types.hpp"

namespace Infra {

// 和 GraphSet 里面一样的，最简单的 CSR backend
template <Config config>
class GlobalMemoryGraph {
  private:
    VIndex_t _v_cnt;
    EIndex_t _e_cnt;
    EIndex_t *_vertexes;  // The start & edge of the edge set
    VIndex_t *_edges;     // The edge set;

    __host__ void build_from_text_file(std::ifstream &file) {
        file >> _v_cnt >> _e_cnt;
        // 读取边
        std::vector<std::vector<VIndex_t>> edges_vec;
        edges_vec.resize(_v_cnt);
        for (EIndex_t i = 0; i < _e_cnt; i++) {
            VIndex_t u, v;
            file >> u >> v;
            if (u == v) {
#ifndef NDEBUG
                std::cerr << "Self loop detected: " << u << std::endl;
#endif
                continue;
            }
            edges_vec[u].push_back(v);
            edges_vec[v].push_back(u);
        }
        // 重新计算边的数量
        _e_cnt = 0;
        for (VIndex_t i = 0; i < _v_cnt; i++) {
            // sort
            std::sort(edges_vec[i].begin(), edges_vec[i].end());
            // unique
            edges_vec[i].erase(
                std::unique(edges_vec[i].begin(), edges_vec[i].end()),
                edges_vec[i].end());
            _e_cnt += edges_vec[i].size();
        }

        // 分配内存
        _vertexes = new EIndex_t[_v_cnt + 1];
        _edges = new VIndex_t[_e_cnt];
        _vertexes[0] = 0;
        for (VIndex_t i = 0; i < _v_cnt; i++) {
            _vertexes[i + 1] = _vertexes[i] + edges_vec[i].size();
            std::copy(edges_vec[i].begin(), edges_vec[i].end(),
                      _edges + _vertexes[i]);
        }
    }

    __host__ void build_from_binary_file(std::ifstream &file) {
        file.read(reinterpret_cast<char *>(&_v_cnt), sizeof(VIndex_t));
        file.read(reinterpret_cast<char *>(&_e_cnt), sizeof(EIndex_t));
        _vertexes = new EIndex_t[_v_cnt + 1];
        _edges = new VIndex_t[_e_cnt];
        file.read(reinterpret_cast<char *>(_vertexes),
                  sizeof(EIndex_t) * (_v_cnt + 1));
        file.read(reinterpret_cast<char *>(_edges), sizeof(VIndex_t) * _e_cnt);
    }

  public:
    __host__ __device__ VIndex_t v_cnt() const { return this->_v_cnt; }
    __host__ __device__ EIndex_t e_cnt() const { return this->_e_cnt; }
    __host__ __device__ EIndex_t *vertexes() const { return this->_vertexes; }
    __host__ __device__ VIndex_t *edges() const { return this->_edges; }
    __host__ __device__ VIndex_t *get_neigh(VIndex_t v) const {
        return _edges + _vertexes[v];
    }
    __host__ __device__ VIndex_t get_neigh_cnt(VIndex_t v) const {
        return _vertexes[v + 1] - _vertexes[v];
    }

    __host__ GlobalMemoryGraph(std::ifstream &file, bool binary) {
        if (binary) {
            build_from_binary_file(file);
        } else {
            build_from_text_file(file);
        }
    }

    __host__ void export_to_binary_file(std::ofstream &file) {
        file.write(reinterpret_cast<char *>(&_v_cnt), sizeof(VIndex_t));
        file.write(reinterpret_cast<char *>(&_e_cnt), sizeof(EIndex_t));
        file.write(reinterpret_cast<char *>(_vertexes),
                   sizeof(EIndex_t) * (_v_cnt + 1));
        file.write(reinterpret_cast<char *>(_edges), sizeof(VIndex_t) * _e_cnt);
    }
};

}  // namespace Infra
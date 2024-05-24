#pragma once
#include <algorithm>
#include <concepts>
#include <fstream>
#include <iostream>
#include <map>
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

        std::map<VIndex_t, VIndex_t> vertex_map;
        std::vector<std::pair<VIndex_t, VIndex_t>> edges;
        for (EIndex_t i = 0; i < _e_cnt; i++) {
            VIndex_t u, v;
            file >> u >> v;

            edges.push_back(std::make_pair(u, v));
            vertex_map[u] = vertex_map[v] = 0;
        }

        // 重新编号
        if (_v_cnt != vertex_map.size()) {
            std::cout << "Vertex Count Mismatch: " << _v_cnt << " vs "
                      << vertex_map.size() << std::endl;
        }
        if (_v_cnt < vertex_map.size()) {
            std::cout << "Vertex Count is less than appeared: " << _v_cnt
                      << " vs " << vertex_map.size() << std::endl;
            exit(1);
        }
        VIndex_t new_index = 0;
        for (auto &it : vertex_map) {
            it.second = new_index++;
        }

        // 读取边
        std::vector<std::vector<VIndex_t>> edges_vec;
        edges_vec.resize(_v_cnt);
        for (EIndex_t i = 0; i < _e_cnt; i++) {
            VIndex_t u, v;
            u = vertex_map[edges[i].first];
            v = vertex_map[edges[i].second];

            if (u == v) {
#ifndef NDEBUG
                std::cout << "Self loop detected: " << u << std::endl;
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
    __host__ void to_device() {
        EIndex_t *d_vertexes;
        VIndex_t *d_edges;
        cudaMalloc(&d_vertexes, sizeof(EIndex_t) * (_v_cnt + 1));
        cudaMalloc(&d_edges, sizeof(VIndex_t) * _e_cnt);
        cudaMemcpy(d_vertexes, _vertexes, sizeof(EIndex_t) * (_v_cnt + 1),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_edges, _edges, sizeof(VIndex_t) * _e_cnt,
                   cudaMemcpyHostToDevice);
        // 释放内存
        delete[] _vertexes;
        delete[] _edges;

        // 更改指针
        _vertexes = d_vertexes;
        _edges = d_edges;
    }

    __host__ __device__ inline VIndex_t v_cnt() const { return this->_v_cnt; }
    __host__ __device__ inline EIndex_t e_cnt() const { return this->_e_cnt; }
    __host__ __device__ inline EIndex_t *vertexes() const {
        return this->_vertexes;
    }
    __host__ __device__ inline VIndex_t *edges() const { return this->_edges; }
    __host__ __device__ inline VIndex_t max_degree() const {
        VIndex_t ans = 0;
        for (int i = 0; i < v_cnt(); i++) {
            ans = max(ans, get_neigh_cnt(i));
        }
        return ans;
    }
    __host__ __device__ inline VIndex_t *get_neigh(VIndex_t v) const {
        return _edges + _vertexes[v];
    }
    __host__ __device__ inline VIndex_t get_neigh_cnt(VIndex_t v) const {
        return _vertexes[v + 1] - _vertexes[v];
    }

    __host__ bool has_edge(VIndex_t u, VIndex_t v) const {
        if (get_neigh_cnt(u) > get_neigh_cnt(v)) {
            std::swap(u, v);
        }
        VIndex_t *neigh = get_neigh(u), *end = get_neigh(u) + get_neigh_cnt(u);
        int *t = std::lower_bound(neigh, end, v);
        return t != end && *t == v;
    }

    __host__ void output() const {
        std::cout << "Vertex Count: " << v_cnt() << std::endl;
        std::cout << "Edge Count: " << e_cnt() << std::endl;
        std::cout << "Max Degree: " << max_degree() << std::endl;
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
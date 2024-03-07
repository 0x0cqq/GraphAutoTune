#pragma once
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "core/graph.cuh"

// 这里先完全 Borrow GraphSet 的已有代码

namespace Core {

class Prefix {
  public:
    static constexpr int MAX_DEPTH = 10;
    VIndex_t data[MAX_DEPTH];  // Prefix 的内容
    int depth;                 // Prefix 拥有的长度

    Prefix(const std::vector<VIndex_t> &_data) {
        assert(_data.size() <= MAX_DEPTH);
        depth = _data.size();
        for (int i = 0; i < depth; i++) {
            data[i] = _data[i];
        }
    }

    bool operator==(const Prefix &other) const {
        if (depth != other.depth) {
            return false;
        }
        for (int i = 0; i < depth; i++) {
            if (data[i] != other.data[i]) {
                return false;
            }
        }
        return true;
    }
    bool operator==(const std::vector<VIndex_t> &other) const {
        if (depth != other.size()) {
            return false;
        }
        for (int i = 0; i < depth; i++) {
            if (data[i] != other[i]) {
                return false;
            }
        }
        return true;
    }
};

// helper functions

Pattern get_permutated_pattern(const std::vector<int> &permutation_order,
                               const Pattern &p) {
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
bool is_pattern_valid(const Pattern &p) {
    // every point(except the first one) must connect to some one previous in the
    // permutation
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

int get_iep_suffix_num(const Pattern &p) {
    // 最后的若干个节点，他们没有相互的依赖。
    // 只需要检验最靠前的节点和后面的所有的节点没有相互的依赖
    for (int k = 2; k <= p.v_cnt() - 2; k++) {
        int new_point = p.v_cnt() - k;
        bool has_edge = false;
        for (int index = new_point + 1; index < p.v_cnt(); new_point++) {
            if (!p.has_edge(new_point, index)) {
                has_edge = true;
                break;
            }
        }
        if (has_edge == true) {
            return k - 1;
        }
    }
    return p.v_cnt() - 2;
};

// s 是邻接矩阵，但是去重，去掉自环，bit 长度为 n * (n - 1) / 2
bool is_connected(int s, int n) {
    // 枚举所有子图
    std::vector<bool> connected(n * n, false);
    // 连接所有的边
    int current_cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (s & (1 << current_cnt)) {
                connected[i * n + j] = true;
                connected[j * n + i] = true;
            }
            current_cnt++;
        }
    }
    // 自己连接
    for (int i = 0; i < n; i++) {
        connected[i * n + i] = true;
    }
    // Floyd 闭包
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                connected[i * n + j] =
                    connected[i * n + j] ||
                    (connected[i * n + k] && connected[k * n + j]);
            }
        }
    }

    // 检查是否联通，只需要检查一个点是否联通剩下的所有节点
    for (int j = 0; j < n; j++) {
        if (!connected[j]) {
            return false;
        }
    }
    return true;
};

// 计算 n 个点的联通图，边数为 偶数/奇数 的数量
std::vector<std::pair<int, int>> calculate_graph_cnt(const int k) {
    assert(k * (k - 1) / 2 < 32);  // avoid too many edges
    std::vector<std::pair<int, int>> graph_cnt(k + 1);

    graph_cnt[1] = {1, 0};

    for (int n = 2; n <= k; n++) {
        int m = n * (n - 1) / 2;
        graph_cnt[n] = {0, 0};
        for (int s = 0; s < (1 << m); s++) {
            if (is_connected(s, n)) {
                if (__builtin_popcount(s) % 2 == 0) {
                    graph_cnt[n].first++;  // 偶数条边
                } else {
                    graph_cnt[n].second++;  // 奇数条边
                }
            }
        }
    }

    return graph_cnt;
};

struct IEPInfo {
    // len(vertex_ids) = the number of distinct prefixs used in the IEP process
    std::vector<int> iep_vertex_id;  // vid -> pid: ver_ids[vid] = prefix_id
    std::vector<int> iep_ans_pos;    // gid -> vid: int -> int, pos[gid] = vid
    std::vector<int> iep_coef;       // subgroup -> coef: int -> int
    std::vector<bool> iep_flag;      // subgroup -> flag: int -> bool

    void output() const {
        for (int vid = 0; vid < iep_vertex_id.size(); vid++) {
            std::cout << "vid: " << vid << " prefix id: " << iep_vertex_id[vid]
                      << std::endl;
        }
        std::cout << "Subgroup info:" << std::endl;

        std::cout << "gid: ";
        for (int gid = 0; gid < iep_ans_pos.size(); gid++) {
            std::cout << gid << "\t";
        }
        std::cout << std::endl;

        std::cout << "vid: ";
        for (int gid = 0; gid < iep_ans_pos.size(); gid++) {
            std::cout << iep_ans_pos[gid] << "\t";
        }
        std::cout << std::endl;

        std::cout << "coef: ";
        for (int gid = 0; gid < iep_coef.size(); gid++) {
            std::cout << iep_coef[gid] << "\t";
        }
        std::cout << std::endl;

        std::cout << "flag: ";
        for (int gid = 0; gid < iep_flag.size(); gid++) {
            std::cout << iep_flag[gid] << "\t";
        }
        std::cout << std::endl;
    }
};

struct IEPGroup {
    std::vector<std::vector<int>> group;
    int coef;
};

struct IEPHelperInfo {
    int suffix_num;  // 有多少个点可以参与 IEP
    std::vector<std::pair<int, int>>
        graph_cnt;  // 点数为 k 的连通图，边数为偶数/奇数的数量
    std::vector<IEPGroup> groups;  // 多个联通块的分割情况和系数
};

void get_iep_groups(int depth, std::vector<int> &id, int group_cnt,
                    IEPHelperInfo &helper_info) {
    const auto &graph_cnt = helper_info.graph_cnt;
    // 边界
    if (depth == helper_info.suffix_num) {
        std::vector<int> group_size(group_cnt, 0);
        for (auto g_id : id) {
            group_size[g_id]++;
        }
        std::pair<int, int> val{graph_cnt[0]};
        for (int i = 1; i < group_cnt; i++) {
            int group_sz = group_size[i];
            int new_first =
                val.first * graph_cnt[group_sz].first +
                val.second * graph_cnt[group_sz].second;  // 偶+偶=偶，奇+奇=偶
            int new_second =
                val.first * graph_cnt[group_sz].second +
                val.second * graph_cnt[group_sz].first;  // 偶+奇=奇，奇+偶=奇
            val = {new_first, new_second};
        }
        // 构建 group
        std::vector<std::vector<int>> group{};
        for (int group_id = 0; group_id < group_cnt; group_id++) {
            std::vector<int> cur;
            for (int v_id = 0; v_id < helper_info.suffix_num; v_id++) {
                if (id[v_id] == group_id) cur.push_back(v_id);
            }
            group.push_back(cur);
        }

        helper_info.groups.emplace_back(
            IEPGroup{group, val.first - val.second});
        return;
    }

    // 递归
    // 分出来一个新类
    id[depth] = group_cnt;
    get_iep_groups(depth + 1, id, group_cnt + 1, helper_info);

    // 仍然分到老的类里面去
    for (int i = 0; i < depth; i++) {
        id[depth] = id[i];
        get_iep_groups(depth + 1, id, group_cnt, helper_info);
    }
}

IEPHelperInfo generate_iep_helper_info(const Pattern &p) {
    int iep_suffix_num = get_iep_suffix_num(p);
    if (iep_suffix_num <= 1) {
        // 无法使用
        return IEPHelperInfo{0, {}, {}};
    } else {
        // 可以使用 IEP, 提前计算一些系数的信息
        IEPHelperInfo helper_info;

        // 单个连通块的情况
        std::vector<std::pair<int, int>> graph_cnt_by_edges =
            calculate_graph_cnt(iep_suffix_num);

        std::vector<VIndex_t> id(iep_suffix_num);

        helper_info.suffix_num = iep_suffix_num;
        helper_info.graph_cnt = graph_cnt_by_edges;

        // 多个连通块的情况
        get_iep_groups(0, id, 0, helper_info);

        return helper_info;
    }
}

class Schedule {
  public:
    int basic_prefix_num;
    int total_prefix_num;
    std::vector<Prefix> prefixs;
    std::vector<int> prefixs_father;
    IEPInfo iep_info;

    int insert_prefix(std::vector<int> &data) {
        if (data.size() == 0) {
            return -1;
        }

        // 自己是否已经出现？
        int pos =
            std::find(prefixs.begin(), prefixs.end(), data) - prefixs.begin();
        if (pos != prefixs.size()) {
            return pos;
        }

        // 否则，剪掉最后一个点，然后插入
        std::vector<int> tmp_data{data.begin(), data.end() - 1};
        int father = insert_prefix(tmp_data);

        // 插入新的点
        prefixs.emplace_back(data);
        prefixs_father.push_back(father);
        return prefixs.size() - 1;
    }

    void calculate_pattern(const Pattern &p) {
        // 正式计算
        // 存在一些没有爹的点，这是不合法的，会导致答案非常多
        if (!is_pattern_valid(p)) {
#ifndef NDEBUG
            std::cerr << "Invalid pattern" << std::endl;
#endif
            return;
        }

        // IEP 优化的辅助信息
        IEPHelperInfo helper_info = generate_iep_helper_info(p);

        // 构建主 Schedule

        // 构建 basic_prefix，也就是不考虑 iep 情况下的 prefix。
        for (int i = 0; i < p.v_cnt(); i++) {
            std::vector<int> tmp_data;
            for (int j = 0; j < i; j++) {
                if (p.has_edge(i, j)) {
                    tmp_data.push_back(j);
                }
            }
            int prefix_id = insert_prefix(tmp_data);
            // what to do with prefix id?
        }

        basic_prefix_num = prefixs.size();

        // 构建 iep_prefix

        for (int rank = 0; rank < helper_info.suffix_num; rank++) {
            const auto &group = helper_info.groups[rank].group;
            const auto &coef = helper_info.groups[rank].coef;

            const VIndex_t suffix_base = p.v_cnt() - helper_info.suffix_num;

            for (const auto &sub_group : group) {
                std::vector<int> tmp_data;
                for (int i = 0; i < suffix_base; i++) {
                    for (int j = 0; j < sub_group.size(); j++) {
                        if (p.has_edge(i, suffix_base + sub_group[j])) {
                            tmp_data.push_back(i);
                            break;
                        }
                    }
                }

                // 这个 iep 的“subgroup”需要的前缀
                int prefix_id = insert_prefix(tmp_data);
                // vertex_ids[vid] = prefix_id
                // ans[vid] = vertex_set[vertex_ids[vid]].size()
                int vid = std::find(iep_info.iep_vertex_id.begin(),
                                    iep_info.iep_vertex_id.end(), prefix_id) -
                          iep_info.iep_vertex_id.begin();
                if (vid == iep_info.iep_vertex_id.size()) {
                    iep_info.iep_vertex_id.push_back(prefix_id);
                    // vid 不变
                }
                iep_info.iep_ans_pos.push_back(vid);

                // 最后一个 group
                if (rank == helper_info.suffix_num - 1) {
                    iep_info.iep_coef.push_back(coef);
                    iep_info.iep_coef.push_back(true);
                } else {
                    iep_info.iep_coef.push_back(0);
                    iep_info.iep_flag.push_back(false);
                }
            }
        }

        total_prefix_num = prefixs.size();

        // 处理 next ？

        // 如果没有 child，就只需要 size

        // 增加限制
        // 目前的限制生成是非常愚蠢的，需要有 matching 的算法才能生成

        // 计算 cost

        // 根据 cost 更新 best val
    }

    void output() const {
        std::cout << "IEP_INFO: " << std::endl;
        iep_info.output();
    }

    Schedule(const Pattern &p) {
        // 只用当前的顺序，不枚举 permutation
        calculate_pattern(p);

        // 最好的 permutation 以及对应的 cost 评估值
        // std::vector<int> best_permutation;
        // double best_cost;

        // std::vector<int> permutation_order;
        // for (int i = 0; i < p.v_cnt(); i++) {
        //     permutation_order.push_back(i);
        // }

        // do {  // 枚举所有的排列
        //     Pattern new_p = get_permutated_pattern(permutation_order, p);

        //     calculate_pattern(new_p);

        // } while (std::next_permutation(permutation_order.begin(),
        //                                permutation_order.end()));
    }
};
}  // namespace Core
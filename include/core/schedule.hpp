#pragma once
#include <algorithm>
#include <cassert>
#include <vector>

#include "core/graph.cuh"

// 这里先完全 Borrow GraphSet 的已有代码

namespace Core {

class Prefix : public std::vector<VIndex_t> {};

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

bool is_pattern_valid(const Pattern &p) {
    // every point must connect to some one previous in the
    // permutation
    for (int i = 0; i < p.v_cnt(); i++) {
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
    // 边数
    int m = n * (n - 1) / 2;
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

struct IEPGroup {
    std::vector<std::vector<int>> group;
    int coef;
};

struct IEPInfo {
    std::vector<IEPGroup> groups;
};

struct IEPHelperInfo {
    int &suffix_num;  // 有多少个点可以参与 IEP
    std::vector<std::pair<int, int>> &graph_cnt;
};

void get_iep_groups(int depth, std::vector<int> &id, int group_cnt,
                    const IEPHelperInfo &helper_info,
                    IEPInfo &result_iep_info) {
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

        result_iep_info.groups.emplace_back(
            IEPGroup{group, val.first - val.second});
        return;
    }

    // 递归
    // 分出来一个新类
    id[depth] = group_cnt;
    get_iep_groups(depth + 1, id, group_cnt + 1, helper_info, result_iep_info);

    // 仍然分到老的类里面去
    for (int i = 0; i < depth; i++) {
        id[depth] = id[i];
        get_iep_groups(depth + 1, id, group_cnt, helper_info, result_iep_info);
    }
};

class Schedule {
  public:
    int basic_prefix_num;
    int total_prefix_num;
    IEPInfo iep_info;

    int find_father_prefix(std::vector<int> &data) {
        if (data.size() == 0) {
            return -1;
        }
        int last_num = data[data.size() - 1];
    }

    Schedule(const Pattern &p) {
        std::vector<int> permutation_order;

        // best ones
        std::vector<int> best_permutation;
        double best_val;

        for (int i = 0; i < p.v_cnt(); i++) {
            permutation_order.push_back(i);
        }
        do {  // 枚举所有的排列

            // 正式计算
            Pattern new_p = get_permutated_pattern(permutation_order, p);

            // 存在一些没有爹的点，这是不合法的
            if (!is_pattern_valid(new_p)) {
                continue;
            }

            // IEP 优化
            int iep_suffix_num = get_iep_suffix_num(new_p);
            if (iep_suffix_num <= 1) {
                // 无法使用 IEP
            } else {
                // 可以使用 IEP
                // 单个连通块的情况
                std::vector<std::pair<int, int>> graph_cnt_by_edges =
                    calculate_graph_cnt(iep_suffix_num);

                std::vector<VIndex_t> id(iep_suffix_num);

                const IEPHelperInfo helper_info{iep_suffix_num,
                                                graph_cnt_by_edges};

                this->iep_info.groups.clear();

                get_iep_groups(0, id, 0, helper_info, this->iep_info);
            }

            // 构建主 Schedule

            // 构建 basic_prefix，也就是不考虑 iep 情况下的 prefix。
            std::vector<int> tmp_data;

        } while (std::next_permutation(permutation_order.begin(),
                                       permutation_order.end()));
    }
};
}  // namespace Core
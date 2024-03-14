#pragma once
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "configs/gpu_consts.cuh"
#include "core/graph.cuh"
#include "core/pattern.hpp"

// 这里先完全 Borrow GraphSet 的已有代码

namespace Core {

class Prefix {
  public:
    VIndex_t data[MAX_DEPTH];  // Prefix 的内容
    int depth;                 // Prefix 拥有的长度

    constexpr Prefix() : depth(0) {}

    constexpr Prefix(const std::vector<VIndex_t> &_data) {
        assert(_data.size() <= MAX_DEPTH);
        depth = _data.size();
        for (int i = 0; i < depth; i++) {
            data[i] = _data[i];
        }
    }

    constexpr bool operator==(const Prefix &other) const {
        if (depth != other.depth) {
            return false;
        }
        return std::equal(data, data + depth, other.data);
    }

    constexpr bool operator==(const std::vector<VIndex_t> &other) const {
        if (depth != other.size()) {
            return false;
        }
        return std::equal(data, data + depth, other.begin());
    }

    void output() const {
        std::cout << "Prefix: ";
        for (int i = 0; i < depth; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
};

constexpr int get_iep_suffix_num(const Pattern &p) {
    // 最后的若干个节点，他们没有相互的依赖。
    // 只需要检验最靠前的节点和后面的所有的节点没有相互的依赖
    for (int k = 2; k <= p.v_cnt() - 2; k++) {
        int new_point = p.v_cnt() - k;
        bool has_edge = false;
        for (int index = new_point + 1; index < p.v_cnt(); index++) {
            if (p.has_edge(new_point, index)) {
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
constexpr bool is_connected(int s, int n) {
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
constexpr std::vector<std::pair<int, int>> calculate_graph_cnt(const int k) {
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

        std::cout << "\tgid: \t";
        for (int gid = 0; gid < iep_ans_pos.size(); gid++) {
            std::cout << gid << "\t";
        }
        std::cout << std::endl;

        std::cout << "\tvid: \t";
        for (int gid = 0; gid < iep_ans_pos.size(); gid++) {
            std::cout << iep_ans_pos[gid] << "\t";
        }
        std::cout << std::endl;

        std::cout << "\tcoef:\t";
        for (int gid = 0; gid < iep_coef.size(); gid++) {
            std::cout << iep_coef[gid] << "\t";
        }
        std::cout << std::endl;

        std::cout << "\tflag:\t";
        for (int gid = 0; gid < iep_flag.size(); gid++) {
            std::cout << iep_flag[gid] << "\t";
        }
        std::cout << std::endl;
    }
};

struct IEPGroup {
    std::vector<std::vector<int>> group;
    int coef;

    void output(bool show_title = false) const {
        if (show_title) {
            std::cout << "IEPGroup: " << std::endl;
        }
        std::cout << "\tcoef: " << coef << std::endl;
        std::cout << "\telem: ";
        for (const auto &sub_group : group) {
            for (const auto &v : sub_group) {
                std::cout << v << " ";
            }
            std::cout << "| ";
        }
        std::cout << std::endl;
    }
};

// 获取所有让 p 的自同构的 permutation
constexpr std::vector<std::vector<int>> get_isomorphism_permutations(
    const Pattern &p) {
    std::vector<std::vector<int>> ans{};
    std::vector<int> perm{};
    for (int i = 0; i < p.v_cnt(); i++) {
        perm.push_back(i);
    }
    // 枚举所有的 permutation
    do {
        Pattern permutated_p = get_permutated_pattern(perm, p);
        if (permutated_p == p) {
            ans.push_back(perm);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    return ans;
}

constexpr int get_isomorphism_multiplicity(const Pattern &p) {
    std::vector<std::vector<int>> isomorphism_permutations =
        get_isomorphism_permutations(p);
    return isomorphism_permutations.size();
}

struct IEPHelperInfo {
    int suffix_num;                // 有多少个点可以参与 IEP
    std::vector<IEPGroup> groups;  // 多个联通块的分割情况和系数

    void output() const {
        std::cout << "IEPHelperInfo: " << std::endl;
        std::cout << "suffix_num: " << suffix_num << std::endl;
        std::cout << "groups: " << std::endl;
        for (const auto &group : groups) {
            group.output();
        }
    }
};

void get_iep_groups(int depth, std::vector<int> &id, int group_cnt,
                    int suffix_num, std::vector<IEPGroup> &groups,
                    const std::vector<std::pair<int, int>> &graph_cnt) {
    // 边界
    if (depth == suffix_num) {
        std::vector<int> group_size(group_cnt, 0);
        for (auto g_id : id) {
            group_size[g_id]++;
        }
        std::pair<int, int> val{1, 0};
        for (int i = 0; i < group_cnt; i++) {
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
            for (int v_id = 0; v_id < suffix_num; v_id++) {
                if (id[v_id] == group_id) cur.push_back(v_id);
            }
            group.push_back(cur);
        }

        groups.emplace_back(IEPGroup{group, val.first - val.second});
        return;
    }

    // 递归
    // 分出来一个新类
    id[depth] = group_cnt;
    get_iep_groups(depth + 1, id, group_cnt + 1, suffix_num, groups, graph_cnt);

    // 仍然分到老的类里面去
    for (int i = 0; i < group_cnt; i++) {
        id[depth] = i;
        get_iep_groups(depth + 1, id, group_cnt, suffix_num, groups, graph_cnt);
    }
}

constexpr IEPHelperInfo generate_iep_helper_info(const Pattern &p) {
    int iep_suffix_num = get_iep_suffix_num(p);
    if (iep_suffix_num <= 1) {
        // 无法使用
        return IEPHelperInfo{0, {}};
    } else {
        // 可以使用 IEP, 提前计算一些系数的信息

        // 单个连通块的情况
        std::vector<std::pair<int, int>> graph_cnt_by_edges =
            calculate_graph_cnt(iep_suffix_num);

        std::vector<VIndex_t> id(iep_suffix_num);

        std::vector<IEPGroup> groups;

        // 多个连通块的情况
        get_iep_groups(0, id, 0, iep_suffix_num, groups, graph_cnt_by_edges);

        return IEPHelperInfo{iep_suffix_num, groups};
    }
}

class Schedule {
  public:
    int basic_prefix_num;
    int total_prefix_num;
    std::vector<Prefix> prefixs;
    std::vector<int> prefixs_father;
    IEPInfo iep_info;

    // restrictions

    // 将一个新的依赖集合插入前缀。
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
        helper_info.output();

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

        for (int rank = 0; rank < helper_info.groups.size(); rank++) {
            const auto &group = helper_info.groups[rank].group;
            const auto &coef = helper_info.groups[rank].coef;

            const VIndex_t suffix_base = p.v_cnt() - helper_info.suffix_num;

            for (int sg_id = 0; sg_id < group.size(); sg_id++) {
                const auto &sub_group = group[sg_id];
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

                // 最后一个 subgroup: 结算！

                if (sg_id == group.size() - 1) {
                    iep_info.iep_coef.push_back(coef);
                    iep_info.iep_flag.push_back(true);
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
        std::cout << "PREFIXS:" << std::endl;
        for (const auto &prefix : prefixs) {
            std::cout << "\t";
            prefix.output();
        }
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

struct IEPData {
    static constexpr int MAX_PREFIXES = 20;
    static constexpr int MAX_IEP_GROUPS = 50;
    int iep_prefix_num;
    int subgroups_num;
    int iep_vertex_id[MAX_PREFIXES];
    int iep_ans_pos[MAX_IEP_GROUPS];
    int iep_coef[MAX_IEP_GROUPS];
    bool iep_flag[MAX_IEP_GROUPS];
    constexpr IEPData(const IEPInfo &info) {
        iep_prefix_num = info.iep_vertex_id.size();
        subgroups_num = info.iep_ans_pos.size();
        for (int i = 0; i < iep_prefix_num; i++) {
            iep_vertex_id[i] = info.iep_vertex_id[i];
        }
        for (int i = 0; i < subgroups_num; i++) {
            iep_ans_pos[i] = info.iep_ans_pos[i];
            iep_coef[i] = info.iep_coef[i];
            iep_flag[i] = info.iep_flag[i];
        }
    }
};

// data class, used for data transmission
struct ScheduleData {
    static constexpr int MAX_PREFIXES = 20;
    int basic_prefix_num;
    int total_prefix_num;
    Prefix prefixes[MAX_PREFIXES];
    int prefix_fathers[MAX_PREFIXES];
    IEPData iep_data;
    constexpr ScheduleData(const Schedule &schedule)
        : basic_prefix_num(schedule.basic_prefix_num),
          total_prefix_num(schedule.total_prefix_num),
          iep_data(schedule.iep_info) {
        for (int i = 0; i < schedule.prefixs.size(); i++) {
            prefixes[i] = schedule.prefixs[i];
            prefix_fathers[i] = schedule.prefixs_father[i];
        }
    }
    constexpr void to_device() const {
        // do nothing here. no pointer
    }
};

}  // namespace Core
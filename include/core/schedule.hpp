#pragma once
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>

#include "consts/general_consts.hpp"
#include "core/graph.cuh"
#include "core/pattern.hpp"

// 这里先完全 Borrow GraphSet 的已有代码

namespace Core {

class Prefix {
  public:
    VIndex_t data[MAX_VERTEXES];  // Prefix 的内容
    int depth;                    // Prefix 拥有的长度

    Prefix() : depth(0) {}

    Prefix(const std::vector<VIndex_t> &_data) {
        assert(_data.size() <= MAX_VERTEXES);
        depth = _data.size();
        for (int i = 0; i < depth; i++) {
            data[i] = _data[i];
        }
    }

    bool operator==(const Prefix &other) const {
        if (depth != other.depth) {
            return false;
        }
        return std::equal(data, data + depth, other.data);
    }

    bool operator==(const std::vector<VIndex_t> &other) const {
        if (depth != other.size()) {
            return false;
        }
        return std::equal(data, data + depth, other.begin());
    }

    void output(int id, int father_id = -1, int size_only = -1) const {
        std::cout << "Prefix " << id << ": ";
        for (int i = 0; i < depth; i++) {
            std::cout << data[i] << " ";
        }
        if (father_id != -1) {
            std::cout << "F: " << father_id;
        }
        if (size_only != -1) {
            std::cout << " OS: " << size_only;
        }
        std::cout << std::endl;
    }
};

using Permutation = std::vector<int>;
using PermutationGroup = std::vector<std::vector<int>>;
using Restrictions = std::vector<std::pair<int, int>>;
using Pairs = std::vector<std::pair<int, int>>;

int get_iep_suffix_vertexes(const Pattern &p) {
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

// 需要一个在完全图中的 naive match 来检测 restriction 的有效性
// restriction = (x, y) 代表 v_x < v_y 必须得到满足。
int naive_match_in_full_graph(
    const Pattern &p, const std::vector<std::pair<int, int>> &restrictions) {
    if (p.v_cnt() >= 10) {
        std::cout
            << "The Pattern is too big, please reduce the v_cnt less than 10"
            << std::endl;
        return -1;
    }
    std::vector<VIndex_t> data(p.v_cnt());
    std::generate(data.begin(), data.end(), [i = 0]() mutable { return i++; });
    int ans = 0;
    do {
        bool flag = std::all_of(
            restrictions.cbegin(), restrictions.cend(),
            [&data](const auto &p) { return data[p.first] < data[p.second]; });
        if (flag) {
            ans++;
        }
    } while (std::next_permutation(data.begin(), data.end()));
    return ans;
}

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

// 获取所有让 p 的自同构的 permutation
std::vector<Permutation> get_isomorphism_perm(const Pattern &p) {
    std::vector<Permutation> ans{};
    Permutation perm{};
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

int get_isomorphism_multiplicity(const Pattern &p) {
    return get_isomorphism_perm(p).size();
}

PermutationGroup calc_perm_group(const Permutation &perm) {
    int size = perm.size();
    bool use[size];
    for (int i = 0; i < size; i++) use[i] = false;
    PermutationGroup res{};
    for (int i = 0; i < size; i++) {
        if (use[i]) continue;

        std::vector<int> tmp_vec{i};
        use[i] = true;
        int x = perm[i];
        // 当还没有 forming 成一个 cycle 的时候
        while (use[x] == false) {
            use[x] = true;
            tmp_vec.push_back(x);
            x = perm[x];
        }
        res.push_back(tmp_vec);
    }
    return res;
}

bool is_dag(const Pairs &base_dag, int size) {
    std::vector<int> degree(size);
    for (const auto &edge : base_dag) {
        degree[edge.second]++;
    }
    std::queue<int> q;
    int visit_cnt = 0;
    for (int i = 0; i < size; i++) {
        if (degree[i] == 0) q.push(i);
    }
    while (!q.empty()) {
        int x = q.front();
        q.pop();
        visit_cnt++;
        for (const auto &p : base_dag) {
            if (p.first == x) {
                --degree[p.second];
                if (degree[p.second] == 0) q.push(p.second);
            }
        }
    }
    return visit_cnt == size;
}

void aggressive_optimize_dfs(Pairs base_dag, std::vector<Permutation> iso_perm,
                             std::vector<PermutationGroup> perm_groups,
                             std::vector<Pairs> &ordered_pairs_vector,
                             int size) {
    std::vector<Permutation> filtered_iso_perm{};
    std::vector<PermutationGroup> filtered_perm_groups{};

    for (int i = 0; i < iso_perm.size(); i++) {
        Pairs test_dag{base_dag};
        const std::vector<int> &iso = iso_perm[i];
        for (const std::pair<int, int> &pair : base_dag) {
            test_dag.push_back(
                std::make_pair(iso[pair.first], iso[pair.second]));
        }
        // is not dag means conflict
        if (is_dag(test_dag, size)) {
            filtered_perm_groups.push_back(perm_groups[i]);
            filtered_iso_perm.push_back(iso_perm[i]);
        }
    }

    iso_perm = filtered_iso_perm;
    perm_groups = filtered_perm_groups;

    if (iso_perm.size() == 1) {
        ordered_pairs_vector.push_back(base_dag);
        return;
    }

    std::pair<int, int> found_pair;
    for (unsigned int i = 0; i < perm_groups.size(); i++) {
        int two_element_number = 0;

        for (const std::vector<int> &v : perm_groups[i]) {
            if (v.size() == 2) {
                found_pair = std::make_pair(v[0], v[1]);
                std::vector<PermutationGroup> next_perm_groups = perm_groups;
                std::vector<Permutation> next_iso_perm = iso_perm;
                Pairs next_base_dag{base_dag};

                next_perm_groups.erase(next_perm_groups.begin() + i);
                next_iso_perm.erase(next_iso_perm.begin() + i);
                assert(found_pair.first < found_pair.second);
                next_base_dag.push_back(found_pair);

                aggressive_optimize_dfs(next_base_dag, next_iso_perm,
                                        next_perm_groups, ordered_pairs_vector,
                                        size);
            }
        }
        if (two_element_number >= 1) {
            break;
        }
    }
}

// 尽可能多的找到限制，但是不一定是正确的
std::vector<Restrictions> aggressive_optimize_get_all_pairs(const Pattern &p) {
    std::vector<Restrictions> ordered_pairs_vector{};

    std::vector<Permutation> iso_perm = get_isomorphism_perm(p);
    std::vector<PermutationGroup> perm_groups{};
    for (const Permutation &v : iso_perm)
        perm_groups.push_back(calc_perm_group(v));

    std::vector<Permutation> filtered_iso_perm{};
    std::vector<PermutationGroup> filtered_perm_groups{};
    Pairs ordered_pairs{};

    // 删除包含一个两元素排列和一些一元素排列的排列组，记录对应的限制
    for (unsigned int i = 0; i < perm_groups.size(); i++) {
        int two_element_number = 0;
        std::pair<int, int> found_pair;
        // 对于每一个置换
        for (const std::vector<int> &v : perm_groups[i]) {
            // 二元置换
            if (v.size() == 2) {
                ++two_element_number;
                found_pair = std::make_pair(v[0], v[1]);
            } else if (v.size() != 1) {  // or >= 3
                two_element_number = -1;
                break;
            }
        }
        // 如果只有一个二元组
        if (two_element_number == 1) {
            ordered_pairs.push_back(found_pair);
            assert(found_pair.first < found_pair.second);
        } else {
            filtered_perm_groups.push_back(perm_groups[i]);
            filtered_iso_perm.push_back(iso_perm[i]);
        }
    }

    aggressive_optimize_dfs(ordered_pairs, filtered_iso_perm,
                            filtered_perm_groups, ordered_pairs_vector,
                            p.v_cnt());

    return ordered_pairs_vector;
}

std::vector<Restrictions> restricts_generate(const Pattern &p) {
    std::vector<Restrictions> restrictions_list =
        aggressive_optimize_get_all_pairs(p);
    // std::cout << "  Aggressive Optimize: " << restrictions_list.size()
    //           << " restrictions found." << std::endl;
    std::vector<Restrictions> result_restrictions_list;
    int ans =
        naive_match_in_full_graph(p, {}) / get_isomorphism_multiplicity(p);
    for (const auto &restrictions : restrictions_list) {
        int cur_ans = naive_match_in_full_graph(p, restrictions);
        // std::cout << "    [";
        // for (const auto &p : restrictions) {
        //     std::cout << "(" << p.first << ", " << p.second << ") ";
        // }
        // std::cout << "]. Ans: " << cur_ans << std::endl;
        if (cur_ans == ans) {
            result_restrictions_list.push_back(restrictions);
        }
    }
    return result_restrictions_list;
}

double estimate_restrictions(const Pattern &p, const Restrictions &restrictions,
                             VIndex_t v_cnt, EIndex_t e_cnt,
                             long long tri_cnt) {
    int max_degree = p.get_max_degree();
    int size = p.v_cnt();
    int iep_suffix_num = get_iep_suffix_vertexes(p);

    // p_size: pow(p0, i) * v_cnt;
    // pp_size: pow(p1, i)

    Restrictions restricts = restrictions;

    double p0 = e_cnt * 1.0 / v_cnt / v_cnt;  // p_size[1];
    double p1 = tri_cnt * 1.0 * v_cnt / e_cnt / e_cnt;

    int restricts_size = restricts.size();
    std::sort(restricts.begin(), restricts.end());

    std::vector<double> sum(restricts_size);
    std::vector<int> tmp(size);
    for (int i = 0; i < size; i++) tmp[i] = i;

    do {
        for (int i = 0; i < restricts_size; i++)
            if (tmp[restricts[i].first] > tmp[restricts[i].second]) {
                sum[i] += 1;
            } else {
                break;
            }
    } while (std::next_permutation(tmp.begin(), tmp.end()));

    double total = 1;
    for (int i = 2; i <= size; i++) total *= i;
    for (int i = 0; i < restricts_size; i++) sum[i] = sum[i] / total;
    for (int i = restricts_size - 1; i > 0; --i) sum[i] /= sum[i - 1];

    std::vector<int> invariant_size[size];
    for (int i = 0; i < size; i++) invariant_size[i].clear();

    double val = 1;
    for (int i = size - iep_suffix_num - 1; i >= 0; --i) {
        // 有几个连接到自己编号之前的边
        int cnt_forward = 0;
        for (int j = 0; j < i; j++) {
            if (p.has_edge(j, i)) {
                ++cnt_forward;
            }
        }

        // 这是第几个？
        int c = cnt_forward;
        for (int j = i - 1; j >= 0; --j) {
            if (p.has_edge(j, i)) {
                invariant_size[j].push_back(c);
                c--;
            }
        }

        for (int j = 0; j < invariant_size[i].size(); j++) {
            if (invariant_size[i][j] > 1) {
                val += v_cnt * p0 * std::pow(p1, invariant_size[i][j] - 2) *
                       std::log2(p0 * v_cnt);
            }
        }
        for (int j = 0; j < restricts_size; j++) {
            if (restricts[j].second == i) val *= sum[j];
        }
        if (i) {
            val *= v_cnt * p0 * std::pow(p1, cnt_forward - 1);
        } else {
            val *= v_cnt;
        }
    }

    return val;
}

struct IEPInfo {
    // len(vertex_ids) = the number of distinct prefixs used in the IEP process
    std::vector<int> iep_vertex_id;  // vid -> pid: ver_ids[vid] = prefix_id
    std::vector<int> iep_ans_pos;    // gid -> vid: int -> int, pos[gid] = vid
    std::vector<int> iep_coef;       // subgroup -> coef: int -> int
    std::vector<bool> iep_flag;      // subgroup -> flag: int -> bool

    void permute(const std::vector<int> &permutation) {
        // 只有 iep_vertex_id 和 vertex id 有关系
        for (int i = 0; i < iep_vertex_id.size(); i++) {
            int origin_prefix_id = iep_vertex_id[i];
            int place = std::find(permutation.begin(), permutation.end(),
                                  origin_prefix_id) -
                        permutation.begin();
            iep_vertex_id[i] = place;
        }
    }

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

struct IEPHelperInfo {
    int iep_suffix_vertexes;       // 有多少个点可以参与 IEP
    std::vector<IEPGroup> groups;  // 多个联通块的分割情况和系数

    void output() const {
        std::cout << "IEPHelperInfo: ---" << std::endl;
        std::cout << "Iep Suffix Vertexes: " << iep_suffix_vertexes
                  << std::endl;
        std::cout << "Groups: " << std::endl;
        for (const auto &group : groups) {
            group.output();
        }
        std::cout << "------------------" << std::endl;
    }
};

void get_iep_groups(int depth, std::vector<int> &id, int group_cnt,
                    int iep_suffix_vertexes, std::vector<IEPGroup> &groups,
                    const std::vector<std::pair<int, int>> &graph_cnt) {
    // 边界
    if (depth == iep_suffix_vertexes) {
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
            for (int v_id = 0; v_id < iep_suffix_vertexes; v_id++) {
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
    get_iep_groups(depth + 1, id, group_cnt + 1, iep_suffix_vertexes, groups,
                   graph_cnt);

    // 仍然分到老的类里面去
    for (int i = 0; i < group_cnt; i++) {
        id[depth] = i;
        get_iep_groups(depth + 1, id, group_cnt, iep_suffix_vertexes, groups,
                       graph_cnt);
    }
}

// 计算 IEP 所需要的系数的信息
IEPHelperInfo generate_iep_helper_info(const Pattern &p) {
    int iep_suffix_vertexes = get_iep_suffix_vertexes(p);

    // 单个连通块的情况
    std::vector<std::pair<int, int>> graph_cnt_by_edges =
        calculate_graph_cnt(iep_suffix_vertexes);

    std::vector<VIndex_t> id(iep_suffix_vertexes);

    std::vector<IEPGroup> groups;

    // 多个连通块的情况
    get_iep_groups(0, id, 0, iep_suffix_vertexes, groups, graph_cnt_by_edges);

    return IEPHelperInfo{iep_suffix_vertexes, groups};
}

class Schedule {
  public:
    // 能使用 IEP 的点的数量
    int iep_suffix_vertexes;
    // 不考虑 IEP 的点的数量
    int basic_vertexes;
    // 所有的前缀
    std::vector<Prefix> prefixs;
    // 第 i 个节点的 loop set 所用的 prefix
    std::vector<int> loop_set_prefix_id;
    // 最后一个是第 i 个节点的 prefixs
    std::vector<int> vertex_prefix_start;
    std::vector<int> prefixs_father;
    std::vector<int> prefixs_size_only;

    // 限制
    std::vector<int> restrictions_start;
    Restrictions restrictions;

    // 和 IEP 有关的信息
    IEPInfo iep_info;

    void sort_prefixs() {
        std::vector<int> permutation(prefixs.size());
        std::generate(permutation.begin(), permutation.end(),
                      [i = 0]() mutable { return i++; });
        std::sort(permutation.begin(), permutation.end(),
                  [&, this](int a, int b) {
                      auto &p_a = this->prefixs[a], p_b = this->prefixs[b];

                      return std::make_pair(p_a.data[p_a.depth - 1], a) <
                             std::make_pair(p_b.data[p_b.depth - 1], b);
                  });

        // shuffle_prefix_father
        std::vector<Prefix> new_prefixs;
        for (int i = 0; i < prefixs.size(); i++) {
            new_prefixs.push_back(prefixs[permutation[i]]);
        }
        prefixs = new_prefixs;

        // shuffle prefixs_father & iep_info according to_the permutation
        std::vector<int> new_prefixs_father(prefixs_father.size());
        for (int i = 0; i < prefixs_father.size(); i++) {
            int father_this_place = prefixs_father[permutation[i]];

            int father_new_place = 0;
            if (father_this_place == -1) {
                father_new_place = -1;
            } else {
                father_new_place =
                    std::find(permutation.begin(), permutation.end(),
                              father_this_place) -
                    permutation.begin();
            }
            new_prefixs_father[i] = father_new_place;
        }
        prefixs_father = new_prefixs_father;

        iep_info.permute(permutation);
    }

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
            std::cout << "Invalid pattern" << std::endl;
#endif
            return;
        }

        // IEP 优化的辅助信息
        IEPHelperInfo helper_info = generate_iep_helper_info(p);
        helper_info.output();

        this->basic_vertexes = p.v_cnt() - helper_info.iep_suffix_vertexes;
        this->iep_suffix_vertexes = helper_info.iep_suffix_vertexes;

        // 构建主 Schedule

        // 构建 basic_prefix，也就是不考虑 iep 情况下的 prefix。
        for (int i = 0; i < this->basic_vertexes; i++) {
            std::vector<int> tmp_data;
            for (int j = 0; j < i; j++) {
                if (p.has_edge(i, j)) {
                    tmp_data.push_back(j);
                }
            }
            int prefix_id = insert_prefix(tmp_data);
        }
        // 构建 iep_prefix
        for (int rank = 0; rank < helper_info.groups.size(); rank++) {
            const auto &group = helper_info.groups[rank].group;
            const auto &coef = helper_info.groups[rank].coef;

            const VIndex_t suffix_base =
                p.v_cnt() - helper_info.iep_suffix_vertexes;

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
    }

    void output() const {
        std::cout << "Basic vertexes:  " << basic_vertexes << std::endl;
        std::cout << "Loop set prefix id: ";
        for (int i = 0; i < basic_vertexes; i++) {
            std::cout << loop_set_prefix_id[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Prefix start: ";
        assert(vertex_prefix_start.size() == basic_vertexes);
        for (int i = 0; i < basic_vertexes; i++) {
            std::cout << vertex_prefix_start[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "PREFIXS:" << std::endl;
        for (int i = 0; i < prefixs.size(); i++) {
            std::cout << "\t";
            prefixs[i].output(i, prefixs_father[i], prefixs_size_only[i]);
        }
        std::cout << "Restrictions start: (" << restrictions_start.size()
                  << ")";
        assert(restrictions_start.size() == basic_vertexes);
        for (int i = 0; i < basic_vertexes; i++) {
            std::cout << restrictions_start[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Restrictions: ";
        for (const auto &p : restrictions) {
            std::cout << "(" << p.first << ", " << p.second << ") ";
        }
        std::cout << std::endl;
        std::cout << "IEP_INFO: " << std::endl;
        iep_info.output();
    }

    void build_loop_invariant(const Pattern &p) {
        // loop_set_prefix_id
        loop_set_prefix_id.clear();
        loop_set_prefix_id.push_back(-1);
        for (int i = 1; i < basic_vertexes; i++) {
            std::vector<int> data;
            for (int j = 0; j < i; j++) {
                if (p.has_edge(j, i)) {
                    data.push_back(j);
                }
            }
            int pos = std::find(prefixs.begin(), prefixs.end(), data) -
                      prefixs.begin();
            if (pos == prefixs.size()) {
                std::cout << "Prefix not found for vertex " << i << ". Error."
                          << std::endl;
                exit(1);
            }
            loop_set_prefix_id.push_back(pos);
        }

        // vertex_prefixs_range
        vertex_prefix_start.clear();
        int current_prefix_id = 0;
        for (int i = 0; i < basic_vertexes; i++) {
            vertex_prefix_start.push_back(current_prefix_id);
            while (current_prefix_id < prefixs.size() &&
                   prefixs[current_prefix_id]
                           .data[prefixs[current_prefix_id].depth - 1] == i) {
                current_prefix_id++;
            }
        }
    }

    void build_restrictions(const Restrictions &restrictions) {
        this->restrictions = restrictions;
        std::sort(this->restrictions.begin(), this->restrictions.end(),
                  [](const auto &a, const auto &b) {
                      if (a.second != b.second) return a.second < b.second;
                      return a.first < b.first;
                  });
        // filter, 去除所有 second >= basic_vertexes 的限制
        auto it = std::remove_if(
            this->restrictions.begin(), this->restrictions.end(),
            [this](const auto &p) { return p.second >= basic_vertexes; });
        this->restrictions.erase(it, this->restrictions.end());

        // 第一个元素 change 了，就要写一下 start
        int current_start = 0;
        for (int i = 0; i < basic_vertexes; i++) {
            while (current_start < basic_vertexes &&
                   this->restrictions[current_start].second < i) {
                current_start++;
            }
            std::cout << "Current start: " << current_start << std::endl;
            std::cout << "second: " << this->restrictions[current_start].second
                      << std::endl;
            restrictions_start.push_back(current_start);
        }
    }

    void build_only_need_size() {
        // 只要没有 child 的话，就只需要 size
        std::vector<int> has_child(prefixs.size());
        assert(prefixs.size() == prefixs_father.size());
        for (int i = 0; i < prefixs.size(); i++) {
            int f = prefixs_father[i];
            if (f != -1) {
                has_child[f] = 1;
            }
        }
        prefixs_size_only.clear();
        prefixs_size_only.resize(prefixs.size());
        for (int i = 0; i < prefixs.size(); i++) {
            prefixs_size_only[i] = has_child[i] == 1 ? 0 : 1;
        }
    }

    Schedule(const Pattern &p) {
        // 最好的 permutation 以及对应的 cost 评估值
        Permutation best_perm;
        Restrictions best_restrictions;
        int best_iep_prefix_num = -1;
        double best_cost = 1e18;
        bool have_best = false;

        Permutation perm;
        for (int i = 0; i < p.v_cnt(); i++) {
            perm.push_back(i);
        }

        // 这个循环里面不会动 Schedule 的成员函数
        do {
            // 枚举所有的排列
            Pattern new_p = get_permutated_pattern(perm, p);

            // 连通性上出问题 e.t.c.
            if (!is_pattern_valid(new_p)) {
                continue;
            }

            // 如果 IEP 的点数不够，那么就不用考虑了
            // 如果 UEP 的点数更多，那么就抹掉之前的
            int iep_suffix_vertexes = get_iep_suffix_vertexes(new_p);
            if (iep_suffix_vertexes < best_iep_prefix_num) {
                continue;
            } else if (iep_suffix_vertexes > best_iep_prefix_num) {
                best_iep_prefix_num = iep_suffix_vertexes;
                best_cost = 1e18;
                have_best = false;
            }

            // std::cout << "Consider permutation with iep sum ["
            //           << iep_suffix_vertexes << "]: ";
            // for (int i = 0; i < perm.size(); i++) {
            //     std::cout << perm[i] << " ";
            // }
            // std::cout << std::endl;

            // 获取所有可能的限制
            std::vector<Restrictions> all_restrictions =
                restricts_generate(new_p);
            for (const auto &restrictions : all_restrictions) {
                double est = estimate_restrictions(new_p, restrictions, 5000,
                                                   1000000, 10000);
                // std::cout << "    Restrictions: [";
                // for (const auto &p : restrictions) {
                //     std::cout << "(" << p.first << ", " << p.second << ") ";
                // }
                // std::cout << "] Estimation: " << est << std::endl;

                if (!have_best || est < best_cost) {
                    have_best = true;
                    // 更新 best
                    best_cost = est;
                    best_perm = perm;
                    best_restrictions = restrictions;
                }
            }

        } while (std::next_permutation(perm.begin(), perm.end()));

        assert(have_best);

        // 枚举完毕，得到最好的排列和限制
        std::cout << "Best Permutation:";
        for (int i = 0; i < best_perm.size(); i++) {
            std::cout << best_perm[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Best Restrictions:";
        for (const auto &p : best_restrictions) {
            std::cout << "(" << p.first << ", " << p.second << ") ";
        }
        std::cout << std::endl;
        std::cout << "Best Cost: " << best_cost << std::endl;

        Pattern new_p = get_permutated_pattern(best_perm, p);

        output_pattern(new_p);

        // 计算相关信息
        calculate_pattern(new_p);
        // 排序所有的前缀
        sort_prefixs();
        // 构建循环不变式
        build_loop_invariant(new_p);
        // 把限制塞进去
        build_restrictions(best_restrictions);
        // TODO: IEP 的 redundancy

        build_only_need_size();
    }
};

struct IEPData {
    constexpr static int MAX_IEP_GROUPS = 50;
    int iep_prefix_num;
    int subgroups_num;
    int iep_vertex_id[MAX_PREFIXS];
    int iep_ans_pos[MAX_IEP_GROUPS];
    int iep_coef[MAX_IEP_GROUPS];
    bool iep_flag[MAX_IEP_GROUPS];
    IEPData(const IEPInfo &info) {
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
    int iep_suffix_vertexes;
    int basic_vertexes;
    int total_prefix_num;
    int total_restrictions_num;
    Prefix prefixs[MAX_PREFIXS];
    int prefixs_father[MAX_PREFIXS];
    int prefixs_size_only[MAX_PREFIXS];
    int vertex_prefix_start[MAX_VERTEXES + 1];  // 这里要包括最后一个
    int loop_set_prefix_id[MAX_VERTEXES];
    int restrictions_start[MAX_VERTEXES + 1];  // 这里要包括最后一个
    int restrictions[MAX_VERTEXES * MAX_VERTEXES];
    IEPData iep_data;
    ScheduleData(const Schedule &schedule)
        : basic_vertexes(schedule.basic_vertexes),
          total_prefix_num(schedule.prefixs.size()),
          iep_suffix_vertexes(schedule.iep_suffix_vertexes),
          iep_data(schedule.iep_info) {
        assert(schedule.prefixs.size() == schedule.prefixs_father.size());
        assert(schedule.prefixs.size() == schedule.prefixs_size_only.size());
        for (int i = 0; i < schedule.prefixs.size(); i++) {
            prefixs[i] = schedule.prefixs[i];
            prefixs_father[i] = schedule.prefixs_father[i];
            prefixs_size_only[i] = schedule.prefixs_size_only[i];
        }
        for (int i = 0; i < basic_vertexes; i++) {
            loop_set_prefix_id[i] = schedule.loop_set_prefix_id[i];
            vertex_prefix_start[i] = schedule.vertex_prefix_start[i];
        }
        vertex_prefix_start[basic_vertexes] = total_prefix_num;

        for (int i = 0; i < schedule.restrictions.size(); i++) {
            restrictions[i] = schedule.restrictions[i].first;
        }
        for (int i = 0; i < basic_vertexes; i++) {
            restrictions_start[i] = schedule.restrictions_start[i];
        }
        restrictions_start[basic_vertexes] = schedule.restrictions.size();
    }
    void to_device() const {
        // do nothing here. no pointer
    }
};

}  // namespace Core
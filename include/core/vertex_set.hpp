#pragma once
#include "core/types.hpp"

template <int size>
class UnorderedVertexSet {
    // 维护目前已经部分匹配的embedding的所有节点。这个集合不会超过pattern的size，通常比较小。
    UnorderedVertexSet();

    void init();

    template <int pos>
    void append(VIndex_t value);

    void remove();

    template <int current_size>
    void has_data(VIndex_t value);
};

class OrderedVertexSet {
    // 维护图中某个节点的邻居集合以及他们的交集。这些集合通常较大，而且可以一直维持集合的有序性。
    OrderedVertexSet();

    void construct();

    void intersect_with(const OrderedVertexSet &other);

    template <int size>
    VIndex_t subtraction_with_size(const UnorderedVertexSet<size> &other);
};
#pragma once

enum VertexStoreType {
    Array,
    Bitmap,
};

enum SetIntersectionType {
    Parallel,
    Serial,
};

struct VertexSetConfig {
    // Search Type 从 Linear 转向 Binary 的 Bar。
    // 特殊情况：0 是全 Binary, INT_MAX 是全 Linear
    int set_search_type = 0;
    SetIntersectionType set_intersection_type = Parallel;
    VertexStoreType vertex_store_type = Array;
};
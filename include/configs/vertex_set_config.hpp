#pragma once

enum VertexStoreType {
    Array,
    Bitmap,
};

enum SetIntersectionType {
    Parallel,
    Sequential,
};

enum SetSearchType {
    Binary,
    Serial,
};

struct VertexSetConfig {
    SetSearchType set_search_type = Binary;
    SetIntersectionType set_intersection_type = Parallel;
    VertexStoreType vertex_store_type = Array;
};
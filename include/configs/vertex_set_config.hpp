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
    VertexStoreType vertex_store_type = Array;
    SetIntersectionType set_intersection_type = Parallel;
    SetSearchType set_search_type = Binary;
};

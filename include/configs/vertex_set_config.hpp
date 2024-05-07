#pragma once

enum VertexStoreType {
    Array,
    Bitmap,
};

enum SetIntersectionType {
    Parallel,
    Serial,
};

enum SetSearchType {
    Binary,
    Linear,
};

struct VertexSetConfig {
    SetSearchType set_search_type = Binary;
    SetIntersectionType set_intersection_type = Parallel;
    VertexStoreType vertex_store_type = Array;
};
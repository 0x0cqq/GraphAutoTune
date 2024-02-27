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
    VertexStoreType vertexStoreType = Array;
    SetIntersectionType setIntersectionType = Parallel;
    SetSearchType setSearchType = Binary;
};
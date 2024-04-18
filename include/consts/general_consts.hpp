#pragma once

constexpr int MAX_VERTEXES = 10;  // 最大的遍历深度
constexpr int MAX_PREFIXS = 10;   // 最多的 Prefix 个数

enum DeviceType {
    CPU_DEVICE,
    GPU_DEVICE,
    UNKNOWN_DEVICE,
};

constexpr int LOG_DEPTH = 0;
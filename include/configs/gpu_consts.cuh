#pragma once

constexpr int THREADS_PER_BLOCK = 128;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

constexpr int num_blocks = 10;
constexpr int num_total_warps = num_blocks * WARPS_PER_BLOCK;

// 最大的遍历深度
constexpr int MAX_VERTEXES = 10;
// 最多的 Prefix 个数
constexpr int MAX_PREFIXS = 10;

enum DeviceType {
    CPU_DEVICE,
    GPU_DEVICE,
    UNKNOWN_DEVICE,
};

constexpr int LOG_DEPTH = 6;
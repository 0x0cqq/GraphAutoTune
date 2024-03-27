#pragma once

constexpr int THREADS_PER_BLOCK = 128;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

constexpr int num_blocks = 10;
constexpr int num_total_warps = num_blocks * WARPS_PER_BLOCK;

// temporary
constexpr int MAX_DEPTH = 10;

enum DeviceType {
    CPU_DEVICE,
    GPU_DEVICE,
    UNKNOWN_DEVICE,
};

constexpr int LOG_DEPTH = 6;
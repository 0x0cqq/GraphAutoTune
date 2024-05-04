#pragma once

// struct LaunchConfig {
//     int num_blocks;
//     int warps_per_block;
//     int threads_per_block;
// };

// TODO，Launch Task 如何确定参数？

namespace LaunchConfig {

constexpr int THREADS_PER_BLOCK = 128;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

constexpr int num_blocks = 512;
constexpr int num_total_warps = num_blocks * WARPS_PER_BLOCK;

}  // namespace LaunchConfig
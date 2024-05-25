#pragma once

struct LaunchConfig {
    int num_blocks = 1024;
    int threads_per_block = 256;
    int threads_per_warp = 32;
    int max_regs = 64;
};

// 与 Engine 相关的配置
struct EngineConfig {
    LaunchConfig launch_config;
};
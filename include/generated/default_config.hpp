
#pragma once
#include "configs/config.hpp"

constexpr Config default_config{.vertex_set_config = {.set_search_type = 32, .set_intersection_type = Parallel, .vertex_store_type = Array}, .infra_config = {.graph_backend_type = InMemory}, .engine_config = {.launch_config = {.num_blocks = 1024, .threads_per_block = 256, .threads_per_warp = 32, .max_regs = 48}}};



#pragma once
#include "configs/config.hpp"

constexpr Config default_config{.vertex_set_config = {.set_search_type = Binary, .set_intersection_type = Parallel, .vertex_store_type = Array}, .infra_config = {.graph_backend_type = InMemory}, .engine_config = {.max_set_size = 10000, .nums_unit = 10000}};



#pragma once
#include "configs/config.hpp"

constexpr Config default_config{.vertex_set_config = {.set_search_type = 16, .set_intersection_type = Serial, .vertex_store_type = Array}, .infra_config = {.graph_backend_type = InMemory}, .engine_config = {}};


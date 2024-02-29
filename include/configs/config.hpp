#pragma once

#include "configs/engine_config.hpp"
#include "configs/graph_config.hpp"
#include "configs/vertex_set_config.hpp"

struct Config {
    VertexSetConfig vertex_set_config;
    GraphConfig graph_config;
    EngineConfig engine_config;
};
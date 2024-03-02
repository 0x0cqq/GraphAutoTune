#pragma once

#include <type_traits>

#include "configs/engine_config.hpp"
#include "configs/graph_config.hpp"
#include "configs/vertex_set_config.hpp"

// 总的 Config，包含了所有的配置
struct Config {
    VertexSetConfig vertex_set_config;
    GraphConfig graph_config;
    EngineConfig engine_config;
};

template <Config config>
class VertexSetTypeDispatcher {
    // 默认占位符
};

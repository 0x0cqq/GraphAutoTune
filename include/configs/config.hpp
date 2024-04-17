#pragma once

#include <type_traits>

#include "configs/engine_config.hpp"
#include "configs/infra_config.hpp"
#include "configs/vertex_set_config.hpp"

// 总的 Config，包含了所有的配置
struct Config {
    VertexSetConfig vertex_set_config;
    EngineConfig engine_config;
    InfraConfig infra_config;
};

// 类型的默认占位符
template <Config config>
class VertexSetTypeDispatcher {};

template <Config config>
class GraphBackendTypeDispatcher {};

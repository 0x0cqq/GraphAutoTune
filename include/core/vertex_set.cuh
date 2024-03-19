#pragma once
#include <concepts>

#include "configs/config.hpp"
#include "core/types.hpp"
#include "utils/utils.hpp"

// implementations
#include "implementations/array_vertex_set.cuh"
#include "implementations/bitmap_vertex_set.cuh"

namespace Core {

template <typename Impl>
concept IsVertexSetImpl = requires(Impl t, VIndex_t *data, VIndex_t size) {
    { t.init(data, size) } -> std::same_as<void>;
    { t.size() } -> std::same_as<VIndex_t>;
    { t.data() } -> std::same_as<VIndex_t *>;
    { t.storage_space() } -> std::same_as<size_t>;
    { t.clear() } -> std::same_as<void>;
    { t.intersect(t, t) } -> std::same_as<void>;
    // TODO: 增加 for each
};

}  // namespace Core

template <Config config>
requires(config.vertex_set_config.vertex_store_type == Array)
class VertexSetTypeDispatcher<config> {
  public:
    using type = GPU::ArrayVertexSet<config>;
};

template <Config config>
requires(config.vertex_set_config.vertex_store_type == Bitmap)
class VertexSetTypeDispatcher<config> {
  public:
    using type = GPU::BitmapVertexSet<config>;
};
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
    { t.__init(data, size) } -> std::same_as<void>;
    { t.__size() } -> std::same_as<VIndex_t>;
    { t.__data() } -> std::same_as<VIndex_t *>;
    { t.__storage_space() } -> std::same_as<size_t>;
    { t.__clear() } -> std::same_as<void>;
    { t.__intersect(t) } -> std::same_as<void>;
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
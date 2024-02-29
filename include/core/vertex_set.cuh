#pragma once
#include <concepts>

#include "configs/config.hpp"
#include "core/types.hpp"
#include "utils/utils.hpp"

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

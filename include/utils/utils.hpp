#pragma once

template <bool x = false>
void assert_false() {
    static_assert(x, "This should never be reached");
}
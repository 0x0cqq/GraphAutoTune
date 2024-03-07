#include <catch2/catch_test_macros.hpp>

#include "core/schedule.hpp"

TEST_CASE("Schedule Test", "[schedule]") {
    Core::Pattern p{"0110010101101010010101010"};

    Core::Schedule schedule{p};

    schedule.output();
}
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "configs/config.hpp"
#include "engine/engine.cuh"
#include "utils/cuda_utils.cuh"

constexpr Config default_config{};

__global__ void test_engine(Engine::Executor<default_config> *engine) {
    // engine->perform_search();
}

TEST_CASE("Engine Test", "[engine test]") {
    Engine::Executor<default_config> *engine;

    std::cout << "Size of the Exector: "
              << sizeof(Engine::Executor<default_config>) << "\n";

    gpuErrchk(cudaMalloc(&engine, sizeof(Engine::Executor<default_config>)));

    size_t free, total;
    gpuErrchk(cudaMemGetInfo(&free, &total));
    int used = int(double(total - free) / (1024 * 1024));
    std::cout << "Used memory: " << used << " MB\n";

    test_engine<<<1, 32>>>(engine);

    gpuErrchk(cudaDeviceSynchronize());
}
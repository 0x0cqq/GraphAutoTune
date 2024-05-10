#include <cooperative_groups.h>

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "configs/config.hpp"
#include "core/vertex_set.cuh"
#include "implementations/array_vertex_set.cuh"
#include "utils/cuda_utils.cuh"

namespace cg = cooperative_groups;

constexpr Config array_config{
    .vertex_set_config = {.vertex_store_type = Array}};

template <typename Impl>
// requires Core::IsVertexSetImpl<Impl>
__global__ void kernel(Impl &vset1, Impl &vset2, VIndex_t *data1,
                       VIndex_t *data2, VIndex_t N, VIndex_t &result) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    vset1.init(warp, data1, N);
    vset2.init(warp, data2, N);

    __syncthreads();
    __threadfence_block();

    vset1.intersect(warp, vset1, vset2);
    __syncthreads();
    __threadfence_block();

    if (threadIdx.x == 0) {
        printf("Intersection: %d\n", vset1.size());
        result = vset1.size();
    }
}

TEST_CASE("Array Vertex Set Test", "[array_vertex_set]") {
    using VertexSet = VertexSetTypeDispatcher<array_config>::type;
    VertexSet *s1, *s2;

    int N = 30;
    VIndex_t *a = new VIndex_t[N], *b = new VIndex_t[N];
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = (i + 1) * 2;
    }

    // allocate memory

    VIndex_t *dev_a = nullptr, *dev_b = nullptr;
    gpuErrchk(cudaMalloc(&dev_a, N * sizeof(VIndex_t)));
    gpuErrchk(cudaMalloc(&dev_b, N * sizeof(VIndex_t)));

    VIndex_t *result;
    gpuErrchk(cudaMallocManaged(&result, sizeof(VIndex_t)));

    // copy data to device
    gpuErrchk(
        cudaMemcpy(dev_a, a, N * sizeof(VIndex_t), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(dev_b, b, N * sizeof(VIndex_t), cudaMemcpyHostToDevice));

    // allocate array_vertex_set
    gpuErrchk(cudaMalloc(&s1, sizeof(VertexSet)));
    gpuErrchk(cudaMalloc(&s2, sizeof(VertexSet)));

    // initialize array_vertex_set

    kernel<<<1, 32>>>(*s1, *s2, dev_a, dev_b, N, *result);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    REQUIRE(*result == 15);

    // free memory

    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(s1));
    gpuErrchk(cudaFree(s2));
    gpuErrchk(cudaFree(result));

    delete[] a;
    delete[] b;
}
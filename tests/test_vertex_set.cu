#include <array>
#include <catch2/catch_test_macros.hpp>

#include "configs/config.hpp"
#include "core/vertex_set.cuh"
#include "implementations/array_vertex_set.cuh"
#include "implementations/bitmap_vertex_set.cuh"
#include "utils/cuda_utils.cuh"

constexpr Config array_config{
    .vertex_set_config = {.vertex_store_type = Array}};

template <typename Impl>
requires Core::IsVertexSetImpl<Impl>
__global__ void kernel(Impl &vset1, Impl &vset2, VIndex_t *data1,
                       VIndex_t *data2, VIndex_t N, VIndex_t &result) {
    vset1.init(data1, N);
    vset2.init(data2, N);

    __syncthreads();
    __threadfence_block();

    vset1.intersect(vset1, vset2);
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

constexpr Config bitmap_config{
    .vertex_set_config = {.vertex_store_type = Bitmap}};

TEST_CASE("Bitmap Vertex Set Test", "[Bitmap_vertex_set]") {
    using VertexSet = VertexSetTypeDispatcher<bitmap_config>::type;
    VertexSet *s1, *s2;

    constexpr int N = 30;
    std::array<VIndex_t, N> a, b;
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = (i + 1) * 2;
    }

    constexpr int M = GPU::get_storage_space<bitmap_config>();

    printf("M: %d\n", M);

    std::array<VIndex_t, M> set_a, set_b;

    GPU::prepare_bitmap_data_cpu<bitmap_config>(a.data(), N, set_a.data());
    GPU::prepare_bitmap_data_cpu<bitmap_config>(b.data(), N, set_b.data());

    // allocate set memory

    VIndex_t *dev_set_a = nullptr, *dev_set_b = nullptr;
    gpuErrchk(cudaMalloc(&dev_set_a, M * sizeof(VIndex_t)));
    gpuErrchk(cudaMemset(dev_set_a, 0, M * sizeof(VIndex_t)));

    gpuErrchk(cudaMalloc(&dev_set_b, M * sizeof(VIndex_t)));
    gpuErrchk(cudaMemset(dev_set_b, 0, M * sizeof(VIndex_t)));

    // copy data to device
    gpuErrchk(cudaMemcpy(dev_set_a, set_a.data(), M * sizeof(VIndex_t),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_set_b, set_b.data(), M * sizeof(VIndex_t),
                         cudaMemcpyHostToDevice));

    VIndex_t *result;
    gpuErrchk(cudaMallocManaged(&result, sizeof(VIndex_t)));

    // allocate array_vertex_set
    gpuErrchk(cudaMalloc(&s1, sizeof(VertexSet)));
    gpuErrchk(cudaMalloc(&s2, sizeof(VertexSet)));

    // initialize array_vertex_set

    kernel<<<1, 32>>>(*s1, *s2, dev_set_a, dev_set_b, M, *result);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    REQUIRE(*result == 15);

    // free memory

    gpuErrchk(cudaFree(dev_set_a));
    gpuErrchk(cudaFree(dev_set_b));
    gpuErrchk(cudaFree(s1));
    gpuErrchk(cudaFree(s2));
    gpuErrchk(cudaFree(result));
}
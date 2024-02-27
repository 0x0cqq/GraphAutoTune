#include <catch2/catch_test_macros.hpp>

#include "configs/config.hpp"
#include "implementations/gpu/array_vertex_set.cuh"
#include "utils/cuda_utils.cuh"

constexpr Config config{{Array, Parallel, Binary}};

template <typename Impl>
requires GPU::IsGPUVertexSetImpl<Impl>
__global__ void kernel(Impl &vset1, Impl &vset2, VIndex_t *data1,
                       VIndex_t *data2, VIndex_t N) {
    if (threadIdx.x == 0) {
        vset1.__init(data1, N);
        vset2.__init(data2, N);
    }
    __syncthreads();
    __threadfence_block();

    vset1.__intersect(vset2);
    __syncthreads();
    __threadfence_block();

    if (threadIdx.x == 0) {
        printf("Intersection: %d\n", vset1.__size());
    }
}

TEST_CASE("Array Vertex Set Test", "[array_vertex_set]") {
    GPU::ArrayVertexSet<config> *s1, *s2;

    int N = 10;
    VIndex_t *a = new VIndex_t[N], *b = new VIndex_t[N];
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = (i + 1) * 2;
    }

    // allocate memory

    VIndex_t *dev_a = nullptr, *dev_b = nullptr;
    gpuErrchk(cudaMalloc(&dev_a, N * sizeof(VIndex_t)));
    gpuErrchk(cudaMalloc(&dev_b, N * sizeof(VIndex_t)));

    // copy data to device
    gpuErrchk(
        cudaMemcpy(dev_a, a, N * sizeof(VIndex_t), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(dev_b, b, N * sizeof(VIndex_t), cudaMemcpyHostToDevice));

    // allocate array_vertex_set
    gpuErrchk(cudaMalloc(&s1, sizeof(GPU::ArrayVertexSet<config>)));
    gpuErrchk(cudaMalloc(&s2, sizeof(GPU::ArrayVertexSet<config>)));

    // initialize array_vertex_set

    kernel<<<1, 32>>>(*s1, *s2, dev_a, dev_b, N);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    // free memory

    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(s1));
    gpuErrchk(cudaFree(s2));
}
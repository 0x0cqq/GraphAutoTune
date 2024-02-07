#include <iostream>

#include "gpu/gpu_vertex_set.cuh"

__global__ void testKernel(GPUVertexSet *vset) {
    printf("Hello from the GPU, vertex_set %d\n", vset->function());
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    GPUVertexSet set{};

    testKernel<<<1, 1>>>(&set);

    cudaDeviceSynchronize();

    return 0;
}

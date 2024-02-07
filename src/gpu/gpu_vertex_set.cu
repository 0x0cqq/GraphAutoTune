#include <iostream>

#include "gpu/gpu_vertex_set.cuh"

GPUVertexSet::GPUVertexSet() {
    std::cout << "GPUVertexSet created" << std::endl;
}

GPUVertexSet::~GPUVertexSet() {
    std::cout << "GPUVertexSet destroyed" << std::endl;
}

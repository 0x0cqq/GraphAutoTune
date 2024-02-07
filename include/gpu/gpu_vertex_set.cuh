#pragma once

class GPUVertexSet {
  public:
    GPUVertexSet();
    ~GPUVertexSet();
    __device__ int function() { return 1; }
};
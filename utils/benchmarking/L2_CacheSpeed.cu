#include <sys/time.h>
#include <time.h>

#include <iostream>
#define USECPSEC 1000000ULL

// find largest power of 2
unsigned flp2(unsigned x) {
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

unsigned long long dtime_usec(unsigned long long start = 0) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

using mt = unsigned long long;
__global__ void k(mt *d, mt *d2, int len, int lps) {
    for (int l = 0; l < lps; l++)
        for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < len;
             i += gridDim.x * blockDim.x)
            d[i] = __ldcg(d2 + i);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int nTPSM = prop.maxThreadsPerMultiProcessor;
    const int nSM = prop.multiProcessorCount;
    const unsigned l2size = prop.l2CacheSize;
    unsigned sz = flp2(l2size) / 2;
    sz = sz / sizeof(mt);  // approx 1/2 the size of the L2
    const int nTPB = 512;  // block size
    const int nBLK = (nSM * nTPSM) / nTPB;
    const int loops = 100;
    mt *d, *d2;
    cudaMalloc(&d, sz * sizeof(mt));
    cudaMalloc(&d2, sz * sizeof(mt));
    k<<<nBLK, nTPB>>>(d, d2, sz, 1);  // warm-up
    cudaDeviceSynchronize();
    unsigned long long dt = dtime_usec(0);
    k<<<nBLK, nTPB>>>(d, d2, sz, loops);
    cudaDeviceSynchronize();
    dt = dtime_usec(dt);
    std::cout << "bw: " << (sz * 2 * sizeof(mt) * loops) / (float)dt << "MB/s"
              << std::endl;
}
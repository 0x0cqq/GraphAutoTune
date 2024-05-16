#pragma once
#include <iomanip>
#include <iostream>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        std::cout << "GPUassert " << code << ": " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

int get_device_information(bool print = false) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (print) std::cout << "Device Information:" << std::endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if (print)
            std::cout << "Device " << i << ", name: " << prop.name << std::endl;
        if (i == 0) {
            if (print)
                std::cout << "  Memory Clock Rate (KHz): "
                          << prop.memoryClockRate << std::endl;
            if (print)
                std::cout << "  Memory Bus Width (bits): "
                          << prop.memoryBusWidth << std::endl;
            if (print) std::cout.precision(2);
            if (print)
                std::cout << "  Peak Memory Bandwidth (GB/s): "
                          << 2.0 * prop.memoryClockRate *
                                 (prop.memoryBusWidth / 8) / 1.0e6
                          << std::endl;
        }
    }
    return nDevices;
}

void print_device_memory() {
    size_t free_byte, used_byte, total_byte;
    gpuErrchk(cudaMemGetInfo(&free_byte, &total_byte));
    used_byte = total_byte - free_byte;

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(2) << "Memory Stats: " << "used: "
              << double(used_byte) / 1024 / 1024 / 1024 << " GB, "
              << "total: " << double(total_byte) / 1024 / 1024 / 1024 << " GB"
              << std::endl;
    std::cout.unsetf(std::ios::fixed);
    std::cout << std::setprecision(6);
}

template <typename T>
__device__ int lower_bound(const T *loop_data_ptr, int loop_size,
                           T min_vertex) {
    int l = 0, r = loop_size - 1;
    while (l <= r) {
        int mid = r - ((r - l) >> 1);
        if (loop_data_ptr[mid] < min_vertex)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return l;
}

template <typename T>
__device__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}
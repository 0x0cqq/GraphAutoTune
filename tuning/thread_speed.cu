#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cub/cub.cuh>
#include <iostream>

#define ull unsigned long long

#define WARP_SIZE 16

void test(ull limit_i, ull limit_j, ull *ans) {
    ull local_ans = 0;
    for (ull i = 1; i < limit_i; i++) {
        for (ull j = i; j < limit_j + i; j += 1) {
            local_ans += 1LL * (i * j);
        }
    }
    *ans = local_ans;
}

void test_omp(ull limit_i, ull limit_j, ull *ans) {
    ull local_ans = 0;
#pragma omp parallel num_threads(32)
    {
#pragma omp for reduction(+ : local_ans)
        for (ull i = 1; i < limit_i; i++) {
            for (ull j = i; j < limit_j + i; j += 1) {
                local_ans += 1LL * (i * j);
            }
        }
    }

    *ans = local_ans;
}

__global__ void test_kernel(ull limit_i, ull limit_j, ull *ans) {
    ull local_ans = 0;
    for (ull i = 1; i < limit_i; i++) {
        for (ull j = i; j < limit_j + i; j += 1) {
            local_ans += 1LL * (i * j);
        }
    }
    *ans = local_ans;
}

__global__ void test_kernel_warp(ull limit_i, ull limit_j, ull *ans) {
    ull tid = threadIdx.x;
    ull block_size = blockDim.x;

    __syncthreads();
    ull local_ans = 0;
    for (ull base = 1; base < limit_i; base += block_size) {
        ull i = base + tid;
        if (i >= limit_i) continue;
        for (ull j = i; j < limit_j + i; j += 1) {
            local_ans += 1LL * (i * j);
        }
    }
    typedef cub::WarpReduce<ull, WARP_SIZE> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;
    ull block_sum = WarpReduce(temp_storage).Sum(local_ans);
    __syncthreads();
    if (tid == 0) {
        *ans = block_sum;
    }
}

__global__ void test_kernel_grid(ull limit_i, ull limit_j, ull *ans) {
    ull bid = blockIdx.x;
    ull tid = threadIdx.x;
    ull block_size = blockDim.x;
    ull grid_size = blockDim.x * gridDim.x;
    ull local_ans = 0;
    for (ull base = 1; base < limit_i; base += grid_size) {
        ull i = base + bid * block_size + tid;
        if (i >= limit_i) continue;
        for (ull j = i; j < limit_j + i; j += 1) {
            local_ans += 1LL * (i * j);
        }
    }
    __syncthreads();
    typedef cub::BlockReduce<ull, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    ull block_sum = BlockReduce(temp_storage).Sum(local_ans);
    if (tid == 0) {
        atomicAdd(ans, block_sum);
    }
}

void test_time(auto f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    std::cout << double(duration) / 1000 << " ms" << std::endl;
}

int main() {
    ull *p;
    ull ans;

    ull limit_i = 100000;
    ull limit_j = 100000;

    cudaMalloc(&p, sizeof(ull));

    test_kernel_warp<<<1, 32>>>(limit_i, limit_j, p);
    cudaDeviceSynchronize();

    // printf("Single thread: ");
    // test_time([&]() {
    //     test_kernel<<<1, 1>>>(limit_i, limit_j, p);
    //     cudaDeviceSynchronize();
    // });
    // cudaMemcpy(&ans, p, sizeof(ull), cudaMemcpyDeviceToHost);
    // printf("%lld\n", ans);

    printf("Warps: ");
    cudaMemset(p, 0, sizeof(ull));
    test_time([&]() {
        test_kernel_warp<<<1, WARP_SIZE>>>(limit_i, limit_j, p);
        cudaDeviceSynchronize();
    });
    cudaMemcpy(&ans, p, sizeof(ull), cudaMemcpyDeviceToHost);
    printf("%lld\n", ans);

    limit_i = 50000;
    limit_j = 50000;

    printf("Warps: ");
    cudaMemset(p, 0, sizeof(ull));
    test_time([&]() {
        test_kernel_warp<<<1, WARP_SIZE>>>(limit_i, limit_j, p);
        cudaDeviceSynchronize();
    });
    cudaMemcpy(&ans, p, sizeof(ull), cudaMemcpyDeviceToHost);
    printf("%lld\n", ans);

    limit_i = 10000;
    limit_j = 10000;

    printf("Warps: ");
    cudaMemset(p, 0, sizeof(ull));
    test_time([&]() {
        test_kernel_warp<<<1, WARP_SIZE>>>(limit_i, limit_j, p);
        cudaDeviceSynchronize();
    });
    cudaMemcpy(&ans, p, sizeof(ull), cudaMemcpyDeviceToHost);
    printf("%lld\n", ans);

    // printf("Grid 128: ");
    // cudaMemset(p, 0, sizeof(ull));
    // test_time([&]() {
    //     test_kernel_grid<<<1, 128>>>(limit_i, limit_j, p);
    //     cudaDeviceSynchronize();
    // });
    // cudaMemcpy(&ans, p, sizeof(ull), cudaMemcpyDeviceToHost);
    // printf("%lld\n", ans);

    // printf("Grid 640: ");
    // cudaMemset(p, 0, sizeof(ull));
    // test_time([&]() {
    //     test_kernel_grid<<<5, 128>>>(limit_i, limit_j, p);
    //     cudaDeviceSynchronize();
    // });
    // cudaMemcpy(&ans, p, sizeof(ull), cudaMemcpyDeviceToHost);
    // printf("%lld\n", ans);

    // printf("Grid 1280: ");
    // cudaMemset(p, 0, sizeof(ull));
    // test_time([&]() {
    //     test_kernel_grid<<<10, 128>>>(limit_i, limit_j, p);
    //     cudaDeviceSynchronize();
    // });
    // cudaMemcpy(&ans, p, sizeof(ull), cudaMemcpyDeviceToHost);
    // printf("%lld\n", ans);

    // printf("CPU: ");
    // test_time([&]() { test(limit_i, limit_j, &ans); });
    // printf("%lld\n", ans);

    return 0;
}
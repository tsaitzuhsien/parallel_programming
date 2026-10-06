#include <stdio.h>
#include <cuda.h>
#include <omp.h>
#include <stdint.h>
#include <time.h>
 
#define LENGTH_MAX 16777216
#define ELEMENT_PER_BLOCK 2048
#define ELEMENT_PER_THREAD 128
 
__device__ inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}
__device__ inline uint32_t encrypt(uint32_t m, uint32_t key) {
    return (rotate_left(m, key&31) + key)^key;
}
 
__global__ void dot_product(uint32_t *C, int n, uint32_t key1, uint32_t key2) {
    int start = blockIdx.x * ELEMENT_PER_BLOCK + threadIdx.x;
 	int end = (blockIdx.x + 1) * ELEMENT_PER_BLOCK;
	end = (end > n)? n:end;
 
	uint32_t sum = 0;
    for (int i = start; i < end; i += blockDim.x) {
        sum += encrypt(i, key1) * encrypt(i, key2);
    }
	int total_idx = blockDim.x * blockIdx.x + threadIdx.x;
	C[total_idx] = sum;
 
    return;
}
 
uint32_t C_host[LENGTH_MAX / ELEMENT_PER_THREAD];
int main() {
    int n;
    uint32_t key1, key2;
 
    while (scanf("%d%u%u", &n, &key1, &key2) == 3) {
        uint32_t *C_device;
		int thread_n = ((n + ELEMENT_PER_BLOCK - 1) / ELEMENT_PER_BLOCK) * (ELEMENT_PER_BLOCK / ELEMENT_PER_THREAD);
        cudaMalloc((void **) &C_device, sizeof(uint32_t) * thread_n);
 
        dim3 block_dim (ELEMENT_PER_BLOCK / ELEMENT_PER_THREAD);
        dim3 grid_dim ((n + ELEMENT_PER_BLOCK - 1) / ELEMENT_PER_BLOCK);
 
        dot_product<<<grid_dim, block_dim>>>(C_device, n, key1, key2);
        cudaMemcpy(C_host, C_device, sizeof(uint32_t) * thread_n, cudaMemcpyDeviceToHost);
 
        uint32_t sum = 0;
#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < thread_n; i++)
            sum += C_host[i];
 
        printf("%u\n", sum);
        cudaFree(C_device);
    }
    return 0;
}

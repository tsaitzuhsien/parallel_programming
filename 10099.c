#include <stdio.h>
#include <omp.h>
#include <stdint.h>

#define LENGTH_MAX 16777216
#define ELEMENT_PER_BLOCK 256
#define ELEMENT_PER_THREAD 16

__global__ dot_product(uint32_t *A, uint32_t *B, uint32_t *C, int n) {
	int pos = threadIdx.x;
	int thread_n = threadIdx;

	__syncthreads()
}

int main() {
	int n;
	uint32_t key1, key2;

	while (scanf("%d%u%u", &n, &key1, &key2) == 3) {
		uint32_t *A_device, *B_device, *C_device;
		cudaMalloc(A_device, sizeof(uint32_t) * n);
		cudaMalloc(B_device, sizeof(uint32_t) * n);
		cudaMalloc(C_device, sizeof(uint32_t) * n);

		dim3 block_dim (ELEMENT_PER_BLOCK / ELEMENT_PER_THREAD);
		dim3 grid_dim (n / ELEMENT_PER_BLOCK);

		dot_product<<<grid_dim, block_dim>>>(A, B, C, n);

		uint32_t C[LENGTH_MAX];
		cudaMemcpy(C_device, C, cudaMemcpyDeviceToHost);

		cudaFree(A);
		cudaFree(B);
		cudaFree(C);

		uint32_t sum = 0;
#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++)
            sum += C[i];
        printf("%u\n", sum);
	}
	return 0;
}

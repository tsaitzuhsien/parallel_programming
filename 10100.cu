#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cuda.h>

#define N_MAX 1024
#define THREAD_PER_BLOCK 1024

void rand_gen(uint32_t seed, int N, uint32_t matrix[]) {
    uint32_t x = 2, n = N * N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + seed + i + j) % n;
            matrix[i * N + j] = x;
        }
    }
}

void print_matrix(int N, uint32_t A[]) {
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[");
        for (int j = 0; j < N; j++)
            fprintf(stderr, " %u", A[i * N + j]);
        fprintf(stderr, " ]\n");
    }
}

uint32_t signature(int N, uint32_t A[]) {
    uint32_t h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i * N + j]) * 2654435761LU;
    }
    return h;
}

__global__ void matrix_multiplication(uint32_t result_device[], uint32_t A_device[], uint32_t B_device[], int n) {
	extern __shared__ uint32_t intermediate[];
	uint32_t *A_shared = intermediate + n;

	int A_row = blockIdx.x;

	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		A_shared[i] = A_device[A_row * n + i];	
		intermediate[i] = 0;
	}
	__syncthreads();

	for (int i = 0; i < n; i++) {
		intermediate[threadIdx.x] += A_shared[i] * B_device[i * n + threadIdx.x];
	}

	__syncthreads();
	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		result_device[A_row * n + i] = intermediate[i];
	}
	return;
}

uint32_t A[N_MAX * N_MAX], B[N_MAX * N_MAX], result[N_MAX * N_MAX];

int main() {
    int n;
    uint32_t seed1, seed2;
    assert(scanf("%d %u %u", &n, &seed1, &seed2) == 3);

	uint32_t *result_device;
	cudaMalloc((void **) &result_device, sizeof(uint32_t) * n * n);

    rand_gen(seed1, n, A);
    rand_gen(seed2, n, B);
	uint32_t *A_device, *B_device;
	cudaMalloc((void **) &A_device, sizeof(uint32_t) * n * n);
	cudaMalloc((void **) &B_device, sizeof(uint32_t) * n * n);
	cudaMemcpy(A_device, A, sizeof(uint32_t) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B, sizeof(uint32_t) * n * n, cudaMemcpyHostToDevice);
	
	dim3 block_dim(n);
	dim3 grid_dim(n);
	matrix_multiplication<<<grid_dim, block_dim, sizeof(uint32_t) * (2 * n)>>>(result_device, A_device, B_device, n);

	cudaMemcpy(result, result_device, sizeof(uint32_t) * n * n, cudaMemcpyDeviceToHost);

#ifdef DEBUG
    print_matrix(n, A);
    print_matrix(n, B);
    print_matrix(n, result);
#endif
    printf("%u\n", signature(n, result));
    return 0;
}

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #define DEBUG

#define N_MAX 16777216

#define ELEMENT_PER_BLOCK 4096
#define ELEMENT_PER_THREAD 4

#define THREAD_PER_BLOCK 256

inline uint32_t encrypt(uint32_t m, uint32_t key) {
    return (m * m + key) % key;
}

__global__ void base_sort(uint32_t *A_device, int N) {
	extern __shared__ uint32_t A_shared[];

	int block_start = blockIdx.x * ELEMENT_PER_BLOCK;
	int block_end = (blockIdx.x + 1) * ELEMENT_PER_BLOCK;
	if (block_end > N)
		block_end = N;
	for (int i = threadIdx.x; i < block_end - block_start; i += blockDim.x)
		A_shared[i] = A_device[i + block_start];
	__syncthreads();

	int thread_start = threadIdx.x * ELEMENT_PER_THREAD;
	int thread_end = thread_start + ELEMENT_PER_THREAD;
	if (thread_end > block_end - block_start)
		thread_end = block_end - block_start;

	bool increasing = (threadIdx.x % 2 == 0)? true:false;
	for (int round = 0; round < ELEMENT_PER_THREAD; round++) {
		for (int i = thread_start; i < thread_end - 1; i++) {
			if (increasing == true && A_shared[i] > A_shared[i + 1]) {
				uint32_t tmp = A_shared[i];
				A_shared[i] = A_shared[i + 1];
				A_shared[i + 1] = tmp;
			} else if (increasing == false && A_shared[i] < A_shared[i + 1]) {
				uint32_t tmp = A_shared[i];
				A_shared[i] = A_shared[i + 1];
				A_shared[i + 1] = tmp;
			}
		}
	}
	__syncthreads();

	for (int i = threadIdx.x; i < block_end - block_start; i += blockDim.x)
		A_device[i + block_start] = A_shared[i];

	return;
}

__global__ void bitonic_sort(uint32_t source[], int N, int merge_base, int distance) {
	int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	bool increasing = ((thread_idx / merge_base) % 2 == 0)? true:false;

	int segment_idx = thread_idx / distance;
	int merge_first = segment_idx * 2 * distance + (thread_idx - segment_idx * distance);
	int merge_last = merge_first + distance;
#ifdef DEBUG
	if (threadIdx.x == 3) {
		printf("Thread idx = %d\n", thread_idx);
		printf("Segment idx = %d\n", segment_idx);
		printf("Merge first = %d\n", merge_first);
		printf("Merge last = %d\n", merge_last);
		printf("------------------\n");
	}
#endif
	
	if ((increasing == true && source[merge_first] > source[merge_last])
			|| (increasing == false && source[merge_first] < source[merge_last])) {
		uint32_t tmp = source[merge_first];
		source[merge_first] = source[merge_last];
		source[merge_last] = tmp;
	}
	__syncthreads();
#ifdef DEBUG
	if (threadIdx.x == 0){
		printf("------- After distance %d sort ---------\n", merge_base);
		for (int i = 0; i < N; i++) {
			printf("%u ", source[i]);
		}			
		printf("\n");
	}
#endif

	return;
}

uint32_t A[N_MAX], A_buf[N_MAX];
int main() {
    int N, K;
    while (scanf("%d %d", &N, &K) == 2) {
#pragma omp parallel for
        for (int i = 0; i < N; i++)
            A[i] = encrypt(i, K);

#ifdef DEBUG
		printf("------- Before base sort ---------\n");
		for (int i = 0; i < N; i++) {
			printf("%u ", A[i]);
		}
		printf("\n");
#endif

		uint32_t *A_device;
		cudaMalloc((void **) &A_device, sizeof(uint32_t) * N * 2);
		cudaMemcpy(A_device, A, sizeof(uint32_t) * N, cudaMemcpyHostToDevice);

		dim3 block_dim(ELEMENT_PER_BLOCK / ELEMENT_PER_THREAD);
		dim3 grid_dim((N + ELEMENT_PER_BLOCK - 1) / ELEMENT_PER_BLOCK);
		base_sort<<<grid_dim, block_dim, sizeof(uint32_t) * ELEMENT_PER_BLOCK>>>(A_device, N);
		cudaDeviceSynchronize();

#ifdef DEBUG
		cudaMemcpy(A, A_device, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);
		printf("------- After base sort ---------\n");
		for (int i = 0; i < N; i++) {
			printf("%u ", A[i]);
		}
		printf("\n");
#endif

		int round = 0;
		for (int merge_base = ELEMENT_PER_THREAD; merge_base < N; merge_base *= 2) {
			dim3 block_dim(THREAD_PER_BLOCK);
			dim3 grid_dim(((N / 2) + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);

			for (int distance = merge_base; distance > 0; distance /= 2) {
				bitonic_sort<<<grid_dim, block_dim>>>(A_device, N, merge_base, distance);
				cudaDeviceSynchronize();
			}

		}
		
#ifdef DEBUG
		cudaMemcpy(A, A_device, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);
		for (int i = 0; i < N; i++) {
			printf("%u ", A[i]);
		}
		printf("\n");
#endif

		cudaMemcpy(A, A_device, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);

		uint32_t sum = 0;
		#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++)
            sum += A[i] * i;
        printf("%u\n", sum);

		cudaFree(A_device);
    }
    return 0;
}

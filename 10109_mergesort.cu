#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N_MAX 16777216

#define ELEMENT_PER_BLOCK 4096
#define ELEMENT_PER_THREAD 16

#define THREAD_PER_BLOCK_MERGE 64

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

	for (int round = 0; round < ELEMENT_PER_THREAD; round++) {
		for (int i = thread_start; i < thread_end - 1; i++) {
			if (A_shared[i] > A_shared[i + 1]) {
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

__global__ void merge_sort(uint32_t source[], uint32_t result[], int N, int merge_base) {
	int block_start = (merge_base * 2) * blockIdx.x * blockDim.x;
	int block_end = block_start + (merge_base * 2) * blockDim.x;
	if (block_end > N)
		block_end = N;

	int thread_start1 = block_start + (2 * merge_base) * threadIdx.x;
	int thread_end1 = block_start + (2 * merge_base) * threadIdx.x + merge_base;
	if (thread_end1 > block_end)
		thread_end1 = block_end;
	int thread_start2 = thread_end1;
	int thread_end2 = thread_end1 + merge_base;
	if (thread_end2 > block_end)
		thread_end2 = block_end;
#ifdef DEBUG
	if (threadIdx.x < 4)
		printf("Thread %d: Merge [%d, %d] and [%d, %d]\n", threadIdx.x, thread_start1, thread_end1, thread_start2, thread_end2);
#endif

	int head1 = thread_start1, head2 = thread_start2;
	for (int i = thread_start1; i < thread_end2; i++) {
#ifdef DEBUG
		if (threadIdx.x == 0) {
			printf("%d %d\n", head1, head2);
		}
#endif
		if (head2 == thread_end2) {
			result[i] = source[head1];
			head1++;
		} else if (head1 == thread_end1) {
			result[i] = source[head2];
			head2++;
		} else if (source[head1] < source[head2]) {
			result[i] = source[head1];
			head1++;
		} else {
			result[i] = source[head2];
			head2++;
		}
	}
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
		cudaMemcpy(A, A_device, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);
#ifdef DEBUG
		printf("------- After base sort ---------\n");
		for (int i = 0; i < N; i++) {
			printf("%u ", A[i]);
		}
		printf("\n");
#endif

		int round = 0;
		for (int merge_base = ELEMENT_PER_THREAD; merge_base < N; merge_base *= 2, round++) {
			dim3 block_dim(THREAD_PER_BLOCK_MERGE);
			dim3 grid_dim((N + THREAD_PER_BLOCK_MERGE * merge_base * 2 - 1) / (THREAD_PER_BLOCK_MERGE * merge_base * 2));

			int offset_source = (round % 2 == 0)? 0:N;
			int offset_result = (round % 2 == 0)? N:0;
			merge_sort<<<grid_dim, block_dim>>>(A_device + offset_source, A_device + offset_result, N, merge_base);
			cudaDeviceSynchronize();

#ifdef DEBUG
			cudaMemcpy(A, A_device + offset_result, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);
			printf("------- After %d round of merge sort ---------\n", round);
			for (int i = 0; i < N; i++) {
				printf("%u ", A[i]);
			}
			printf("\n");
#endif
		}

		int offset = (round % 2 == 0)? 0:N;
		cudaMemcpy(A, A_device + offset, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);

		uint32_t sum = 0;
		#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++)
            sum += A[i] * i;
        printf("%u\n", sum);
		cudaFree(A_device);
    }
    return 0;
}

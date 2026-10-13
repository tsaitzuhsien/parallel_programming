#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

__device__ static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}
__device__ static inline uint32_t encrypt(uint32_t m, uint32_t key) {
    return (rotate_left(m, key&31) + key)^key;
}

#define N_MAX 1000000

typedef struct {
	int r, c;
	uint32_t v;
} Cell;

int cmp (const void *x, const void *y) {
	Cell *front = (Cell *)x;
	Cell *rear = (Cell *)y;
	if ( (front->r < rear->r) || (front->r == rear->r && front->c < rear->c) )
		return -1;
	else
		return 1;
}

__global__ void SpMM (const Cell *__restrict__ A_device,
					const Cell *__restrict__ B_device,
					uint32_t *result_device,
					int *offset_A_device,
					int *offset_B_device,
					int nA,
					int nB,
					int diff_nA,
					int diff_nB) {

	extern __shared__ uint32_t hash_sum[];

	const int row_A = blockIdx.x;

	hash_sum[threadIdx.x] = 0;
	for (int row_B = threadIdx.x, tail_A = offset_A_device[row_A + 1]; row_B < diff_nB; row_B += blockDim.x) {
		int head_A = offset_A_device[row_A];
		int head_B = offset_B_device[row_B], tail_B = offset_B_device[row_B + 1];	

		int exact_row = A_device[head_A].r;
		int exact_col = B_device[head_B].r;
		uint32_t current_sum = 0;
		while (head_A < tail_A && head_B < tail_B) {
			if (A_device[head_A].c == B_device[head_B].c) {
				current_sum += A_device[head_A].v * B_device[head_B].v;
				head_A++;
				head_B++;
			} else if (A_device[head_A].c > B_device[head_B].c) {
				head_B++;
			} else {
				head_A++;
			}
		}

		if (current_sum != 0){
			hash_sum[threadIdx.x] += encrypt((exact_row + 1) * (exact_col + 1), current_sum);
		}
	}

	__syncthreads();
	if (threadIdx.x == 0) {
		uint32_t tmp = 0;
		for (int i = 0; i < blockDim.x; i++) {
			tmp += hash_sum[i];
		}
		result_device[blockIdx.x] = tmp;
	}
	
	return;
}

Cell A[N_MAX], B[N_MAX];
uint32_t result[N_MAX];
int offset_A[N_MAX + 1], offset_B[N_MAX + 1];

int main() {
    int N, M, R, nA, nB;
    assert(scanf("%d %d %d", &N, &M, &R) == 3);
    assert(scanf("%d %d", &nA, &nB) == 2);

    for (int i = 0; i < nA; i++)
        assert(scanf("%d %d %d", &A[i].r, &A[i].c, &A[i].v) == 3);
    for (int i = 0; i < nB; i++)
        assert(scanf("%d %d %d", &B[i].c, &B[i].r, &B[i].v) == 3);

	qsort(B, nB, sizeof(Cell), cmp);

	int diff_nA = 0;
	for (int i = 0, current_v = -1; i < nA; i++) {
		if (A[i].r != current_v) {
			offset_A[diff_nA] = i;
			diff_nA++;
			current_v = A[i].r;
		}
	}
	offset_A[diff_nA] = nA;
	
	int diff_nB = 0;
	for (int i = 0, current_v = -1; i < nB; i++) {
		if (B[i].r != current_v) {
			offset_B[diff_nB] = i;
			diff_nB++;
			current_v = B[i].r;
		}
	}
	offset_B[diff_nB] = nB;

	Cell *A_device, *B_device;
	CUDA_CHECK(cudaMalloc((void **) &A_device, sizeof(Cell) * nA));
	CUDA_CHECK(cudaMalloc((void **) &B_device, sizeof(Cell) * nB));
	CUDA_CHECK(cudaMemcpy(A_device, A, sizeof(Cell) * nA, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(B_device, B, sizeof(Cell) * nB, cudaMemcpyHostToDevice));

	uint32_t *result_device;
	CUDA_CHECK(cudaMalloc((void **) &result_device, sizeof(uint32_t) * diff_nA));

	int *offset_A_device, *offset_B_device;
	CUDA_CHECK(cudaMalloc((void **) &offset_A_device, sizeof(int) * (diff_nA + 1)));
	CUDA_CHECK(cudaMalloc((void **) &offset_B_device, sizeof(int) * (diff_nB + 1)));
	CUDA_CHECK(cudaMemcpy(offset_A_device, offset_A, sizeof(int) * (diff_nA + 1), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(offset_B_device, offset_B, sizeof(int) * (diff_nB + 1), cudaMemcpyHostToDevice));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	dim3 block_dim(prop.maxThreadsPerBlock);
	dim3 grid_dim(diff_nA);
	
	SpMM<<<grid_dim, block_dim, sizeof(uint32_t) * prop.maxThreadsPerBlock>>>(A_device, B_device, result_device, offset_A_device, offset_B_device, nA, nB, diff_nA, diff_nB);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	
	CUDA_CHECK(cudaMemcpy(result, result_device, sizeof(uint32_t) * diff_nA, cudaMemcpyDeviceToHost));

	uint32_t hash = 0;
#pragma omp parallel for reduction(+:hash)
	for (int i = 0; i < diff_nA; i++) {
		hash += result[i];
	}
	printf("%u\n", hash);

    return 0;
}

#include <stdio.h>
#include <cuda.h>

#define N_MAX 2000

#define ROW_PER_BLOCK 1
#define COL_PER_THREAD 16

__global__ void game_of_life(int *grid_device, int n, int m) {
	int start_row = blockIdx.x * n * ROW_PER_BLOCK;
	int end_row = start_row + ROW_PER_BLOCK;
	end_row = (end_row > n)? n:end_row;

	for (int row = start_row; row < end_row; row++) {
		
	}

	return;
}

int grid[N_MAX * N_MAX];
int main() {
	int n, period;
	while (scanf("%d%d", &n, &period) == 2) {
		char row[N_MAX];
		for (int i = 0; i < n; i++) {
			scanf("%s\n", row);
			for (int j = 0; j < n; j++) {
				grid[i * n + j] = (row[j] == '1')? 1:0;
			}
		}

		int *grid_device;
		cudaMalloc((void **) &grid_device, sizeof(int) * 2 * n * n);
		cudaMemcpy(grid_device, grid, sizeof(int) * n * n, cudaMemcpyHostToDevice);

		dim3 block_dim((n + COL_PER_BLOCK - 1) / COL_PER_BLOCK);
		dim3 grid_dim((n + ROW_PER_BLOCK - 1) / ROW_PER_BLOCK);

		game_of_life<<<grid_dim, block_dim>>>(grid_device, n, m);
		
		int offset = (period % 2 == 1)? (n * n):0;
		cudaMemcpy(grid, grid_device + offset, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("%d", grid[i * n + j]);
			}
			printf("\n");
		}
	}
	return 0;
}

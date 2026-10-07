#include <stdio.h>
#include <cuda.h>

#define N_MAX 2000

#define ROW_PER_BLOCK 1
#define COL_PER_THREAD 16

__global__ void game_of_life(int *grid_device, int *result_device, int n) {
	int start_row = 1 + blockIdx.x * ROW_PER_BLOCK;
	int end_row = start_row + ROW_PER_BLOCK;
	end_row = (end_row > (n + 1))? (n + 1):end_row;

	int step_r[8] = {1, 1, 0, -1, -1, -1, 0, 1};
	int step_c[8] = {0, -1, -1, -1, 0, 1, 1, 1};

	for (int row = start_row; row < end_row; row++) {
		for (int col = 1 + threadIdx.x; col <= n; col += blockDim.x){
			int alive_neighbor = 0;
			for (int step = 0; step < 8; step++) {
				alive_neighbor += grid_device[(row + step_r[step]) * (n + 2) + col + step_c[step]];
			}

			if (grid_device[row * (n + 2) + col] == 0) {
				if (alive_neighbor == 3) 
					result_device[row * (n + 2) + col] = 1;
				else
					result_device[row * (n + 2) + col] = 0;
			} else {
				if (alive_neighbor == 2 || alive_neighbor == 3)
					result_device[row * (n + 2) + col] = 1;
				else
					result_device[row * (n + 2) + col] = 0;
			}
		}
	}

	return;
}

int grid[(N_MAX + 2) * (N_MAX + 2)];
int main() {
	int n, period;
	while (scanf("%d%d", &n, &period) == 2) {
		char row[N_MAX];
		for (int i = 1; i <= n; i++) {
			scanf("%s\n", row);
			for (int j = 1; j <= n; j++) {
				grid[i * (n + 2) + j] = (row[j - 1] == '1')? 1:0;
			}
		}
#ifdef DEBUG
		printf("------- INPUT --------\n");
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {
				printf("%d", grid[i * (n + 2) + j]);
			}
			printf("\n");
		}
		printf("----------------------\n");
#endif
		int *grid_device;
		cudaMalloc((void **) &grid_device, sizeof(int) * 2 * (n + 2) * (n + 2));
		cudaMemcpy(grid_device, grid, sizeof(int) * (n + 2) * (n + 2), cudaMemcpyHostToDevice);

		dim3 block_dim((n + COL_PER_THREAD - 1) / COL_PER_THREAD);
		dim3 grid_dim((n + ROW_PER_BLOCK - 1) / ROW_PER_BLOCK);

		for (int t = 0; t < period; t++) {
			int prev_offset = (t % 2 == 0)? 0:(n + 2) * (n + 2);
			int current_offset = (t % 2 == 0)? (n + 2) * (n + 2):0;
			game_of_life<<<grid_dim, block_dim>>>(grid_device + prev_offset, grid_device + current_offset, n);
			cudaDeviceSynchronize();
		}
		
		
		int offset = (period % 2 == 0)? 0:((n + 2) * (n + 2));
		cudaMemcpy(grid, grid_device + offset, sizeof(int) * (n + 2) * (n + 2), cudaMemcpyDeviceToHost);

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {
				printf("%d", grid[i * (n + 2) + j]);
			}
			printf("\n");
		}
	}
	return 0;
}

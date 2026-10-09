#include <stdio.h>
#include <omp.h>

#define N_MAX 2000

void game_of_life(int grid_device[][N_MAX + 2], int result_device[][N_MAX + 2], int n) {
	int step_r[8] = {1, 1, 0, -1, -1, -1, 0, 1};
	int step_c[8] = {0, -1, -1, -1, 0, 1, 1, 1};

#pragma omp parallel for schedule(static, 10)
	for (int row = 1; row < n + 1; row++) {
		for (int col = 1; col < n + 1; col++){
			int alive_neighbor = 0;
			for (int step = 0; step < 8; step++) {
				alive_neighbor += grid_device[row + step_r[step]][col + step_c[step]];
			}

			if (grid_device[row][col] == 0) {
				if (alive_neighbor == 3) 
					result_device[row][col] = 1;
				else
					result_device[row][col] = 0;
			} else {
				if (alive_neighbor == 2 || alive_neighbor == 3)
					result_device[row][col] = 1;
				else
					result_device[row][col] = 0;
			}
		}
	}

	return;
}

int grid[2][N_MAX + 2][N_MAX + 2];
int main() {
	int n, period;
	while (scanf("%d%d", &n, &period) == 2) {
		char row[N_MAX];
		for (int i = 1; i <= n; i++) {
			scanf("%s\n", row);
			for (int j = 1; j <= n; j++) {
				grid[0][i][j] = (row[j - 1] == '1')? 1:0;
			}
		}

		for (int t = 0; t < period; t++) {
			game_of_life(grid[t % 2], grid[(t + 1) % 2], n);
		}

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {
				printf("%d", grid[period % 2][i][j]);
			}
			printf("\n");
		}
	}

	return 0;
}

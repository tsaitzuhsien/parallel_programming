#include <stdio.h>
#include <omp.h>
#include <assert.h>

#define N_MAX 1001

char grid[2][N_MAX][N_MAX];

void simulate(int grid_length, int period) {
	for (int t = 0; t < period; t++) {
#pragma omp parallel for
		for (int r = 0; r < grid_length; r++) {
			for (int c = 0; c < grid_length; c++) {
				grid[(t + 1) % 2][r][c] = 'W';
			}
		}

#pragma omp parallel for
		for (int r = 0; r < grid_length; r++) {
			for (int c = 0; c < grid_length; c++) {
				if (grid[t % 2][r][c] == 'R') {
					if (grid[t % 2][r][(c + 1) % grid_length] == 'W') {
						grid[(t + 1) % 2][r][(c + 1) % grid_length] = 'R';
					} else {
						grid[(t + 1) % 2][r][c] = 'R';
					}
				}
			}
		}

#pragma omp parallel for
		for (int r = 0; r < grid_length; r++) {
			for (int c = 0; c < grid_length; c++) {
				if (grid[t % 2][r][c] == 'B') {
					if (grid[(t + 1) % 2][(r + 1) % grid_length][c] == 'W' && grid[t % 2][(r + 1) % grid_length][c] != 'B') {
						grid[(t + 1) % 2][(r + 1) % grid_length][c] = 'B';
					} else {
						grid[(t + 1) % 2][r][c] = 'B';
					}
				}
			}
		}

	} /* period */
	return;
}

int main() {
	int grid_length, period;
	assert(scanf("%d%d", &grid_length, &period) == 2);
	
	for (int r = 0; r < grid_length; r++) {
		assert(scanf("%s\n", grid[0][r]) == 1);
	}

	simulate(grid_length, period);

	for (int r = 0; r < grid_length; r++) {
		grid[period % 2][r][grid_length] = '\0';
		printf("%s\n", grid[period % 2][r]);
	}

	return 0;
}

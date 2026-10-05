#include <stdio.h>
#include <limits.h>

#define N_MAX 4096

long long DP[N_MAX][N_MAX], matrix_size[N_MAX + 1];

int main() {
	int N;
	while (scanf("%d", &N) == 1) {
		for (int i = 0; i <= N; i++){ 
			scanf("%lld", &matrix_size[i]);
		}

#pragma omp parallel for
		for (int i = 0; i < N; i++) {
			DP[i][i] = 0;
		}
			
		for (int len = 2; len <= N; len++) {
#pragma omp parallel for
			for (int i = 0; i <= N - len; i++) {
				long long current_best = LONG_MAX;
				for (int cut = i + 1; cut < i + len; cut++) {
					long long t = DP[i][cut - 1] + DP[i + len - 1][cut]
							+ matrix_size[i] * matrix_size[cut] * matrix_size[i + len];
					if (current_best > t)
						current_best = t;
				}
			   	DP[i][i + len - 1] = current_best;
				DP[i + len - 1][i] = current_best;
			}
		}

		printf("%lld\n", DP[0][N - 1]);
	}

	return 0;
}

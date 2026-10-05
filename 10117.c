#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MAXN 1024

int g[MAXN][MAXN];

int main() {
    int n;
    while (scanf("%d", &n) == 1) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                scanf("%d", &g[i][j]);
                if (g[i][j] == 0)
                    g[i][j] = INT_MAX / 3;
            }
        }
        for (int k = 0; k < n; k++) {
		#pragma omp parallel for
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int cost = g[i][k] + g[k][j];
                    if (g[i][j] > cost)
                        g[i][j] = cost;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d%c", ((g[i][j] == INT_MAX / 3)? 0:g[i][j]), (j == n - 1)? '\n':' ');
            }
        }
    }
    return 0;
}

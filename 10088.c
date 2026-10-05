#include <stdio.h>
#include <omp.h>
#include <stdint.h>
#include "utils.h"

#define N_MAX 10000000
#define THREADS_N 4

uint32_t prefix_sum[N_MAX];

int main() {
	int n;
	uint32_t key;
	omp_set_num_threads(THREADS_N);
	while (scanf("%d %u", &n, &key) == 2) {
        uint32_t sum = 0;
		int base = n / THREADS_N;
#pragma omp parallel
		{
#pragma omp sections
			{
#pragma omp section
				{
					uint32_t sum = 0;
					for (int i = 1; i <= base; i++) {
            			sum += encrypt(i, key);
            			prefix_sum[i] = sum;
					}
				} /* section */
#pragma omp section
				{
					uint32_t sum = 0;
					for (int i = base + 1; i <= 2 * base; i++) {
            			sum += encrypt(i, key);
            			prefix_sum[i] = sum;
					}
				} /* section */
#pragma omp section
				{
					uint32_t sum = 0;
					for (int i = 2 * base + 1; i <= 3 * base; i++) {
            			sum += encrypt(i, key);
            			prefix_sum[i] = sum;
					}
				} /* section */
#pragma omp section
				{
					uint32_t sum = 0;
					for (int i = 3 * base + 1; i <= n; i++) {
            			sum += encrypt(i, key);
            			prefix_sum[i] = sum;
					}
				} /* section */
			} /* sections */
#pragma omp for
			for (int i = base + 1; i <= 2 * base; i++)
				prefix_sum[i] += prefix_sum[base];
#pragma omp for
			for (int i = 2 * base + 1; i <= 3 * base; i++)
				prefix_sum[i] += prefix_sum[2 * base];
#pragma omp for
			for (int i = 3 * base + 1; i <= n; i++)
				prefix_sum[i] += prefix_sum[3 * base];
		} /* parallel */

#ifdef DEBUG
		for (int i = 1; i <= n; i++)
			printf("%d%c", prefix_sum[i], (i == n)?'\n':' ');
#endif

        output(prefix_sum, n);
    }
    return 0;
}

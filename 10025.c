#include <stdio.h>
#include <omp.h>
#include <limits.h>

int main() {
	int AH, AW, BH, BW;
	while(scanf("%d %d %d %d", &AH, &AW, &BH, &BW) != EOF) {

		int A[500][500], B[500][500];
		for (int h = 0; h < AH; h++) {
			for (int w = 0; w < AW; w++) {
				scanf("%d", &A[h][w]);
			}
		}
		for (int h = 0; h < BH; h++) {
			for (int w = 0; w < BW; w++) {
				scanf("%d", &B[h][w]);
			}
		}

		int smallest = INT_MAX;
		int smallest_coordinate[2];
		#pragma omp parallel for
		for (int h = 0; h < AH - BH + 1; h++) {
			for (int w = 0; w < AW - BW + 1; w++) {
				int sum = 0;
				for (int i = 0; i < BH; i++) {
					for (int j = 0; j < BW; j++) {
						sum += (A[h + i][w + j] - B[i][j]) * (A[h + i][w + j] - B[i][j]);
					}
				}
				#pragma omp critical
				if (sum < smallest) {
					smallest = sum;
					smallest_coordinate[0] = h + 1;
					smallest_coordinate[1] = w + 1;
				}
			}
		}
		printf("%d %d\n", smallest_coordinate[0], smallest_coordinate[1]);
	}
	return 0;
}

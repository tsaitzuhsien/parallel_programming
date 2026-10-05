#include <stdio.h>
#include <string.h>

#define STRING_MAX 60000

int DP[3][STRING_MAX];

int main() {
	char A[STRING_MAX], B[STRING_MAX];
	while (scanf("%s%s", A, B) == 2) {
		int A_len = strlen(A), B_len = strlen(B);	

		int long_string = (A_len > B_len)? A_len:B_len;
#pragma omp parallel for
		for (int i = 0; i <= A_len; i++){
			DP[0][i] = 0;
			DP[1][i] = 0;
			DP[2][i] = 0;
		}

		for (int sum = 2; sum <= A_len + B_len; sum++) {
			int start = (sum > B_len)? (sum - B_len):1;
			int end = (sum > A_len)? A_len:(sum - 1);
#pragma omp parallel for
			for (int a = start; a <= end; a++) {
				int b = sum - a;
				int current_best = 0;
				if (A[a - 1] == B[b - 1])
					current_best = DP[(sum - 2) % 3][a - 1] + 1;
				if (DP[(sum - 1) % 3][a] > current_best)
					current_best = DP[(sum - 1) % 3][a];
				if (DP[(sum - 1) % 3][a - 1] > current_best)
					current_best = DP[(sum - 1) % 3][a - 1];
				DP[sum % 3][a] = current_best;
			}
		}

		printf("%d\n", DP[(A_len + B_len) % 3][A_len]);
	}
	return 0;
}

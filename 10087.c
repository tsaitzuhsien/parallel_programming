#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

// #define DEBUG

#define MAXN 1000000

static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
	return  (x << n) | (x >> (32 - n));
}
static inline uint32_t encrypt(uint32_t m, uint32_t key) {
	return (rotate_left(m, key & 31) + key) ^ key;
}

typedef struct {
	int r, c;
	uint32_t value;
} COO;

int cmp (const void *a, const void *b) {
	COO *ptr1 = (COO *)a;
	COO *ptr2 = (COO *)b;
	if (ptr1->r != ptr2->r)
		return ptr1->r > ptr2->r;
	else
		return ptr1->c > ptr2->c;
}

COO A[MAXN], B[MAXN];

void SpMM (int M, int L, int N, int NA, int NB){
	static int offsetA[MAXN], offsetB[MAXN];
	int offsetA_n = 0, offsetB_n = 0;
	for (int i = 0, prev_value = -1; i < NA; i++) {
		if (A[i].r != prev_value) {
			prev_value = A[i].r;
			offsetA[offsetA_n] = i;
			offsetA_n++;
		}
	}
	for (int i = 0, prev_value = -1; i < NB; i++) {
		if (B[i].r != prev_value) {
			prev_value = B[i].r;
			offsetB[offsetB_n] = i;
			offsetB_n++;
		}
	}

	offsetA[offsetA_n] = NA;
	offsetB[offsetB_n] = NB;

	uint32_t hash = 0;
#pragma omp parallel for reduction(+ : hash)
	for (int rA = 0; rA < offsetA_n; rA++) {
		for (int rB = 0; rB < offsetB_n; rB++) {
			int idxA = offsetA[rA], endA = offsetA[rA + 1];
			int idxB = offsetB[rB], endB = offsetB[rB + 1];
			uint32_t value = 0;
			int rC = A[idxA].r, cC = B[idxB].r;
			while (idxA < endA && idxB < endB) {
				if (A[idxA].c == B[idxB].c) {
					value += A[idxA].value * B[idxB].value;
					idxA++;
					idxB++;
				} else if (A[idxA].c < B[idxB].c) {
					idxA++;
				} else {
					idxB++;
				}
			}
			if (value > 0) {
#ifdef DEBUG
				printf("(%d %d) = %u\n", rC, cC, value);
#endif
				hash += encrypt((rC + 1) * (cC + 1), value);
			}
		}	
	}

	printf("%u\n", hash);
	return;
}

int main() {
	int M, L, N;
	assert(scanf("%d%d%d", &M, &L, &N) == 3);

	int NA, NB;
	assert(scanf("%d %d", &NA, &NB) == 2);
	for (int i = 0; i < NA; i++) 
	   assert(scanf("%d %d %d", &A[i].r, &A[i].c, &A[i].value) == 3);
	for (int i = 0; i < NB; i++)
	   assert(scanf("%d %d %d", &B[i].c, &B[i].r, &B[i].value) == 3);
 
	qsort(B, NB, sizeof(COO), cmp);
#ifdef DEBUG
	for (int i = 0; i < NA; i++) 
	   printf("(%d, %d) = %d\n", A[i].r, A[i].c, A[i].value);
	for (int i = 0; i < NB; i++)
	   printf("(%d, %d) = %d\n", B[i].r, B[i].c, B[i].value);
	printf("--------------------\n");
#endif
	SpMM(M, L, N, NA, NB);

	return 0;
}

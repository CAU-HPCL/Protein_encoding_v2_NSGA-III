#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define TOTAL_CODON_NUM 64
#define STOP_CODON_NUM 3
#define CODON_SIZE 3
#define MAX_SYN_CODONS_NUM 6
#define EMPTY -1


/*
Constant memory (64KB)
* CAI 값 계산을 위한 코돈의 weight 값 : float type 64 
* CBP 값 계산을 위한 코돈쌍의 CPS 값 : float type (64 - 3) * (64 - 3)
    CPS [][] 앞 코돈 index 에 뒤 코돈 index 더해서 인덱스 값 구하도록 하기
* 아미노산 번호 값에 따른 코돈의 시작 위치 인덱스 값 : char type (20 + 1) >> 시작위치 * 3 + 동의 코돈 번호 * 3 하면 코돈 정보 얻기 가능
* 코돈 정보 : char type (64 * 3(Codon size) + 1)
* 코돈의 동의 코돈 개수 : char type (20 + 1)
*/

/* Constant memory variables */
__constant__ char c_codons_start_idx[21];
__constant__ char c_syn_codons_num[21];
__constant__ char c_codons[TOTAL_CODON_NUM * CODON_SIZE + 1];
__constant__ float c_codons_weight[TOTAL_CODON_NUM];
__constant__ float c_cps[(TOTAL_CODON_NUM - STOP_CODON_NUM) * (TOTAL_CODON_NUM - STOP_CODON_NUM)];
__constant__ int c_amino_seq_len;
__constant__ int c_cds_len;
__constant__ int c_cds_num;
// __constant__ int c_sort_popsize;
// __constant__ float c_mprob;



/*
Shared memory
Solution : 기존 solution 은 global memory 에 있으니까, shared memory 에는 global memory로 solution 을 가져와서 변이시키고 다시 global memory
에 저장하면 되기 때문에 solution 은 1개만 할당해도 충분
Target protein 아미노산 번호 서열 값
Divide & Conquer 에 필요한 메모리
*/

/*
Sorting 에 필요한 것
Non-dominated sorting 기존거 사용하면 되고
Reference based sorting 하면
Reference point 저장할 공간
solution 이 어떤 point 랑 association 되고 해당 point 와의 거리를 저장할 곳
Normalization
*/

#endif
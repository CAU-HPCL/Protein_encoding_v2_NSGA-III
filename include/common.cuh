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

#define OBJECTIVE_NUM 6
#define MIN_CAI_IDX 0
#define MIN_CBP_IDX 1
#define MIN_HSC_IDX 2
#define MIN_HD_IDX 3
#define MAX_GC_IDX 4
#define MAX_SL_IDX 5

#define RANDOM_GEN 0
#define HIGHEST_CAI_GEN 1

#define MIN_HAIRPIN_DISTANCE 4
#define P 0
#define Q 1
#define L 2

#define PROCEED 1
#define TERMINATION -1

#define GC_UP 1
#define GC_DOWN -1

/* Global memory variables*/
__device__ bool N_cut_check;
__device__ bool HYP_EXCEPTION;
__device__ int d_cur_cycle_num;
__device__ int rank_count;
__device__ int cur_front;
__device__ int g_mutex;
__device__ int number_of_count;
__device__ float f_precision = 0.000001f;
__device__ float estimated_ideal_value[OBJECTIVE_NUM];
__device__ float estimated_worst_value[OBJECTIVE_NUM];
__device__ float estimated_nadir_value[OBJECTIVE_NUM];
__device__ float extreme_points[OBJECTIVE_NUM][OBJECTIVE_NUM];
__device__ float weight_vector[OBJECTIVE_NUM];
__device__ float AB[OBJECTIVE_NUM * (OBJECTIVE_NUM + 1)];


/* Constant memory variables */
__constant__ char c_cds_num;
__constant__ char c_codons_start_idx[21];
__constant__ char c_syn_codons_num[21];
__constant__ char c_codons[TOTAL_CODON_NUM * CODON_SIZE + 1];
__constant__ int c_N; 
__constant__ int c_amino_seq_len;
__constant__ int c_solution_len;
__constant__ int c_cds_len;
__constant__ int c_gen_cycle_num;
__constant__ int c_ref_points_num;
__constant__ float c_cps[(TOTAL_CODON_NUM - STOP_CODON_NUM) * (TOTAL_CODON_NUM - STOP_CODON_NUM)];
__constant__ float c_codons_weight[TOTAL_CODON_NUM];
__constant__ float c_mutation_prob;
__constant__ float c_ref_GC_percent;
__constant__ float c_ref_GC3_percent;

#endif
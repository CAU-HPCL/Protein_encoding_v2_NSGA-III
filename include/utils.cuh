/*
1. Caculating function of Objective functions values
*/

#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__host__ int findAminoIndex(const char amino_abbreviation);

__device__ char findIndexAmongSynonymousCodons(const char *cur_codon, const char *syn_codons, const char syn_codons_num);

__device__ bool isStopCodon(const char *codon);

__device__ char countLeftSideHSC(const char *left_codon, const char *cur_codon);

__device__ char countBothSidesHSC(const char *left_codon, const char *cur_codon, const char *right_codon);

__device__ void indexArrayShuffling(curandStateXORWOW *random_generator, char* index_array, const char array_size);

__device__ char countCodonGC(const char *codon);


__device__ void genPopulation(const thread_block tb, curandStateXORWOW *random_generator, const char *s_amino_seq_idx, char *s_solution, float *s_obj_val, char *s_obj_idx, const char gen_type, float *s_obj_buffer);

#endif

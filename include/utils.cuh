/*
1. Caculating function of Objective functions values
*/

#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>

__device__ char findIndexAmongSynonymousCodons(const char *cur_codon, const char *syn_codons, const char syn_codons_num);

__device__ bool isStopCodon(const char *codon);

__device__ char countLeftSideHSC(const char *left_codon, const char *cur_codon);

__device__ char countBothSidesHSC(const char *left_codon, const char *cur_codon, const char *right_codon);

__device__ void indexArrayShuffling(curandStateXORWOW *random_generator, char* index_array, const char array_size);

__device__ char countCodonGC(const char *codon);

#endif

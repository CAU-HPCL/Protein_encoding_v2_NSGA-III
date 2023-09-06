/*
1. Mutation functions of Objectvie functions
    1.1 Select the best synonymous codon
    1.2 Select random codon among synonymous codons better than current codon
    1.3 Select random codon amsmon synonymous codon
*/

#ifndef MUTATION_H
#define MUTATION_H

#include <curand_kernel.h>

#define SELECT_UPPER 0
#define SELECT_UPPER_RANDOM 1
#define SELECT_RANDOM 2
#define SELECT_HIGH_GC 3
#define SELECT_LOW_GC 4


/* Selecting random codon except current codon */
__device__ void mutationRandom(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx);


/*
1. SELECT_UPPER : Selecting the highest weight codon
2. SELECT_UPPER_RANDOM : Selecting codon among having high weight codons than current codon weight
3. SELECT_RANDOM : Selecting Selecting random codon except current codon
*/
__device__ void mutationCAI(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx, const char mutation_type);


/*
1. SELECT_UPPER : Selecting the highest codon's CPS sum
2. SELECT_UPPER_RANDOM : Selecting codon among having high codons's CPS sum than current codon's CPS sum
3. SELECT_RANDOM : Selecting Selecting random codon except current codon
*/
__device__ void mutationCBP(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char left_aminoacid_idx, const char aminoacid_idx, const char right_aminoacid_idx, const char mutation_type);


/*
1. SELECT_UPPER : Selecting the highest HSC sum
2. SELECT_UPPER_RANDOM : Selecting codon among having high HSC sum than current HSC sum
3. SELECT_RANDOM : Selecting Selecting random codon except current codon
If there are multiple good codon, one codon is selected randomly
*/
__device__ void mutationHSC(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx, const char mutation_type);


/*
This is same as MOBOA & MaOMPE
codon_idx is current codon start point assumming that CDS is one
*/
__device__ void mutationHD(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int codon_idx, const char aminoacid_idx, const char cds1_idx, const char cds2_idx);


/*
1. SELECT_HIGH_GC : Selecting randomly high GC contents than current codon
2. SELECT_LOW_GC : Selecting randomly low GC contents than current codon
3. SELECT_RANDOM : Selecting Selecting random codon except current codon
input mutationp prob is computed before call this function
*/
__device__ void mutationGC(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const char solution_idx, const char aminoacid_idx, const char mutation_type);


#endif
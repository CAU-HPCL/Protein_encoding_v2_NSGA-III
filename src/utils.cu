/*
1. Caculating function of Objective functions values
*/

#include <stdio.h>
#include "../include/common.cuh"


__device__ char findIndexAmongSynonymousCodons(const char *cur_codon, const char *syn_codons, const char syn_codons_num)
{
    for (char i = 0; i < syn_codons_num; i++)
    {
        if (cur_codon[0] == syn_codons[i * CODON_SIZE] && cur_codon[1] == syn_codons[i * CODON_SIZE + 1] && cur_codon[2] == syn_codons[i * CODON_SIZE + 2])
        {
            return i;
        }
    }

    printf("findIndexAmongSynonymousCodons Function failure\n");
}

__device__ char countBothSidesHSC(const char *left_codon, const char *cur_codon, const char *right_codon)
{
    char result = 0;

    

    return result;
}
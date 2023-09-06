/*
1. Caculating function of Objective functions values
*/

#include <stdio.h>

#include <curand_kernel.h>

#include "../include/common.cuh"

__device__ char findIndexAmongSynonymousCodons(const char *cur_codon, const char *syn_codons, const char syn_codons_num)
{
    for (char i = 0; i < syn_codons_num; i++)
    {
        if ((cur_codon[0] == syn_codons[i * CODON_SIZE]) && (cur_codon[1] == syn_codons[i * CODON_SIZE + 1]) && (cur_codon[2] == syn_codons[i * CODON_SIZE + 2]))
        {
            return i;
        }
    }

    printf("findIndexAmongSynonymousCodons Function failure\n");
}

__device__ bool isStopCodon(const char *codon)
{
    for (char i = 1; i <= STOP_CODON_NUM; i++)
    {
        if ((codon[0] == c_codons[(TOTAL_CODON_NUM - i) * CODON_SIZE]) && (codon[1] == c_codons[(TOTAL_CODON_NUM - i) * CODON_SIZE + 1]) && (codon[2] == c_codons[(TOTAL_CODON_NUM - i) * CODON_SIZE + 2]))
        {
            return true;
        }
    }

    return false;
}

__device__ char countLeftSideHSC(const char *left_codon, const char *cur_codon)
{
    char result = 0;

    char temp_codon[3];
    temp_codon[0] = left_codon[1];
    temp_codon[1] = left_codon[2];
    temp_codon[2] = cur_codon[0];
    if (isStopCodon(temp_codon))
    {
        result++;
    }

    temp_codon[0] = left_codon[2];
    temp_codon[1] = cur_codon[0];
    temp_codon[2] = cur_codon[1];
    if (isStopCodon(temp_codon))
    {
        result++;
    }

    return result;
}

__device__ char countBothSidesHSC(const char *left_codon, const char *cur_codon, const char *right_codon)
{
    char result = 0;

    char temp_codon[3];
    temp_codon[0] = left_codon[1];
    temp_codon[1] = left_codon[2];
    temp_codon[2] = cur_codon[0];
    if (isStopCodon(temp_codon))
    {
        result++;
    }

    temp_codon[0] = left_codon[2];
    temp_codon[1] = cur_codon[0];
    temp_codon[2] = cur_codon[1];
    if (isStopCodon(temp_codon))
    {
        result++;
    }

    temp_codon[0] = cur_codon[1];
    temp_codon[1] = cur_codon[2];
    temp_codon[2] = right_codon[0];
    if (isStopCodon(temp_codon))
    {
        result++;
    }

    temp_codon[0] = cur_codon[2];
    temp_codon[1] = right_codon[0];
    temp_codon[2] = right_codon[1];
    if (isStopCodon(temp_codon))
    {
        result++;
    }

    return result;
}

__device__ void indexArrayShuffling(curandStateXORWOW *random_generator, char *index_array, const char array_size)
{
    for (char i = array_size - 1; i > 0; i--)
    {
        char j = (char)(curand_uniform(random_generator) * (i + 1));
        while (j == i + 1)
        {
            j = (char)(curand_uniform(random_generator) * (i + 1));
        }
        char temp = index_array[i];
        index_array[i] = index_array[j];
        index_array[j] = temp;
    }

    return;
}

__device__ char countCodonGC(const char *codon)
{
    char result = 0;

    for (char i = 0; i < CODON_SIZE; i++)
    {
        if (codon[i] == 'G' || codon[i] == 'C')
        {
            result += 1;
        }
    }

    return result;
}

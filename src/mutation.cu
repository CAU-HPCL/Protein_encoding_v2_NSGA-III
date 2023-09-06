/*
1. Mutation functions of Objectvie functions
    1.1 Select the best synonymous codon
    1.2 Select random codon among synonymous codons better than current codon
    1.3 Select random codon amsmon synonymous codon
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "../include/common.cuh"
#include "../include/utils.cuh"

#define SELECT_UPPER 0
#define SELECT_UPPER_RANDOM 1
#define SELECT_RANDOM 2

/* seleceting random codon except current codon */
__device__ void mutationRandom(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
        return;

    char new_codon_idx;
    char cur_codon_idx;

    cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);

    new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
    while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1))
    {
        new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
    }

    if (new_codon_idx == cur_codon_idx)
        new_codon_idx++;

    solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
    solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
    solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];

    return;
}

__device__ void mutationCAI(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx, const char mutation_type)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
        return;

    char new_codon_idx;
    char cur_codon_idx;

    cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);

    switch (mutation_type)
    {
    case SELECT_UPPER:
        solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE];
        solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE + 1];
        solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE + 2];
        break;

    case SELECT_UPPER_RANDOM:
        if (cur_codon_idx != c_syn_codons_num[aminoacid_idx] - 1)
        {
            new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1 - cur_codon_idx));
            while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1 - cur_codon_idx))
            {
                new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1 - cur_codon_idx));
            }
            solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx + 1) * CODON_SIZE];
            solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx + 1) * CODON_SIZE + 1];
            solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx + 1) * CODON_SIZE + 2];
        }
        break;

    case SELECT_RANDOM:
        new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
        while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1))
        {
            new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
        }

        if (new_codon_idx == cur_codon_idx)
            new_codon_idx++;

        solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
        solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
        solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
        break;
    }

    return;
}

__device__ void mutationCBP(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char left_aminoacid_idx, const char aminoacid_idx, const char right_aminoacid_idx, const char mutation_type)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1) || (aminoacid_idx == 20))
        return;

    char new_codon_idx;

    char left_codon_idx;
    char cur_codon_idx;
    char right_codon_idx;

    char upper_cps_sum_codons_idx[MAX_SYN_CODONS_NUM];
    char max_cps_sum_codon_idx;
    char upper_cps_sum_codons_cnt = 0;

    float cur_cps_sum = 0.f;
    float max_cps_sum;

    left_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx - CODON_SIZE], &c_codons[c_codons_start_idx[left_aminoacid_idx] * CODON_SIZE], c_syn_codons_num[left_aminoacid_idx]);
    cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);
    right_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx + CODON_SIZE], &c_codons[c_codons_start_idx[right_aminoacid_idx] * CODON_SIZE], c_syn_codons_num[right_aminoacid_idx]);

    cur_cps_sum += c_cps[(c_codons_start_idx[left_aminoacid_idx] + left_codon_idx) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[aminoacid_idx] + cur_codon_idx]; // left
    if (right_codon_idx < (TOTAL_CODON_NUM - STOP_CODON_NUM))
    {
        cur_cps_sum += c_cps[(c_codons_start_idx[aminoacid_idx] + cur_codon_idx) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[right_aminoacid_idx] + right_codon_idx]; // right
    }

    max_cps_sum = cur_cps_sum;
    for (char i = 0; i < c_syn_codons_num[aminoacid_idx]; i++)
    {
        float new_cps_sum = 0.f;

        if (i != cur_codon_idx)
        {
            new_cps_sum += c_cps[(c_codons_start_idx[left_aminoacid_idx] + left_codon_idx) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[aminoacid_idx] + i]; // left
            if (right_codon_idx < (TOTAL_CODON_NUM - STOP_CODON_NUM))
            {
                cur_cps_sum += c_cps[(c_codons_start_idx[aminoacid_idx] + i) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[right_aminoacid_idx] + right_codon_idx]; // right
            }

            if (new_cps_sum >= cur_cps_sum)
            {
                upper_cps_sum_codons_idx[upper_cps_sum_codons_cnt++] = i;
                if (new_cps_sum > max_cps_sum)
                    max_cps_sum_codon_idx = i;
            }
        }
    }

    switch (mutation_type)
    {
    case SELECT_UPPER:
        if (upper_cps_sum_codons_cnt > 0)
        {
            solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + max_cps_sum_codon_idx) * CODON_SIZE];
            solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + max_cps_sum_codon_idx) * CODON_SIZE + 1];
            solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + max_cps_sum_codon_idx) * CODON_SIZE + 2];
        }
        break;

    case SELECT_UPPER_RANDOM:
        if (upper_cps_sum_codons_cnt > 0)
        {
            new_codon_idx = (char)(curand_uniform(random_generator) * upper_cps_sum_codons_cnt);
            while (new_codon_idx == upper_cps_sum_codons_cnt)
            {
                new_codon_idx = (char)(curand_uniform(random_generator) * upper_cps_sum_codons_cnt);
            }
            solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + upper_cps_sum_codons_idx[new_codon_idx]) * CODON_SIZE];
            solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + upper_cps_sum_codons_idx[new_codon_idx]) * CODON_SIZE + 1];
            solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + upper_cps_sum_codons_idx[new_codon_idx]) * CODON_SIZE + 2];
        }
        break;

    case SELECT_RANDOM:
        new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
        while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1))
        {
            new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
        }

        if (new_codon_idx == cur_codon_idx)
            new_codon_idx++;

        solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
        solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
        solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
        break;
    }

    return;
}

__device__ void mutationHSC(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx, const char mutation_type)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1) || (aminoacid_idx == 20))
        return;

    char new_codon_idx;

    char cur_codon_idx;

    char upper_hsc_sum_codons_idx[MAX_SYN_CODONS_NUM];
    char max_hsc_sum_codon_idx;
    char upper_hsc_sum_codons_cnt = 0;

    char cur_hsc_sum = 0;
    char max_hsc_sum;

    cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);

    // count hidden stop codon number
    cur_hsc_sum = countBothSidesHSC(&solution[solution_idx - CODON_SIZE], &solution[solution_idx], &solution[solution_idx + CODON_SIZE]);

    max_hsc_sum = cur_hsc_sum;
    for (char i = 0; i < c_syn_codons_num[aminoacid_idx]; i++)
    {
        char new_hsc_sum = 0;

        if (i != cur_codon_idx)
        {
            new_hsc_sum = countBothSidesHSC(&solution[solution_idx - CODON_SIZE], &c_codons[(c_codons_start_idx[aminoacid_idx] + i) * CODON_SIZE], &solution[solution_idx + CODON_SIZE]);
            if (new_hsc_sum >= cur_hsc_sum)
            {
                upper_hsc_sum_codons_idx[upper_hsc_sum_codons_cnt++] = i;
                if (new_hsc_sum > max_hsc_sum)
                    max_hsc_sum_codon_idx = i;
            }
        }
    }

    switch (mutation_type)
    {
    case SELECT_UPPER:
        if (upper_hsc_sum_codons_cnt > 0)
        {
            solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + max_hsc_sum_codon_idx) * CODON_SIZE];
            solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + max_hsc_sum_codon_idx) * CODON_SIZE + 1];
            solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + max_hsc_sum_codon_idx) * CODON_SIZE + 2];
        }
        break;

    case SELECT_UPPER_RANDOM:
        if (upper_hsc_sum_codons_cnt > 0)
        {
            new_codon_idx = (char)(curand_uniform(random_generator) * upper_hsc_sum_codons_cnt);
            while (new_codon_idx == upper_hsc_sum_codons_cnt)
            {
                new_codon_idx = (char)(curand_uniform(random_generator) * upper_hsc_sum_codons_cnt);
            }
            solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + upper_hsc_sum_codons_idx[new_codon_idx]) * CODON_SIZE];
            solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + upper_hsc_sum_codons_idx[new_codon_idx]) * CODON_SIZE + 1];
            solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + upper_hsc_sum_codons_idx[new_codon_idx]) * CODON_SIZE + 2];
        }
        break;

    case SELECT_RANDOM:
        new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
        while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1))
        {
            new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
        }

        if (new_codon_idx == cur_codon_idx)
            new_codon_idx++;

        solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
        solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
        solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
        break;
    }

    return;
}

__device__ void mutationHD(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx)
{
    float gen_prob;
    char new_codon_idx;
    char cur_codon_idx;

    cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx]], c_syn_codons_num[aminoacid_idx]);

    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if (gen_prob <= mutation_prob && c_syn_codons_num[aminoacid_idx] != 1)
    {
        switch (mutation_type)
        {
        case SELECT_UPPER:

            break;

        case SELECT_UPPER_RANDOM:

            break;

        case SELECT_RANDOM:
            new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
            while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1))
            {
                new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
            }

            if (new_codon_idx == cur_codon_idx)
                new_codon_idx++;

            solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
            solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
            solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
            break;
        }
    }

    return;
}

__device__ void mutationGC(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx)
{
    return;
}

__device__ void mutationSL(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx)
{
    return;
}
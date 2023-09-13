/*
1. Mutation functions of Objectvie functions
    1.1 Select the best synonymous codon
    1.2 Select random codon among synonymous codons better than current codon
    1.3 Select random codon amsmon synonymous codon
*/

#ifndef MUTATION_H
#define MUTATION_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../include/common.cuh"
#include "../include/utils.cuh"

#define SELECT_UPPER 0
#define SELECT_UPPER_RANDOM 1
#define SELECT_RANDOM 2
#define SELECT_HIGH_GC 3
#define SELECT_LOW_GC 4

/* Selecting random codon except current codon */
__device__ void mutationRandom(const thread_block tb, curandStateXORWOW *random_generator, char *solution, const char *s_amino_seq_idx, const char *s_obj_idx, const float mutation_prob = c_mutation_prob)
{
    int partition_num;
    int idx;
    int amino_seq_idx;
    int solution_idx;
    char aminoacid_idx;
    float gen_prob;

    partition_num = ((c_amino_seq_len * c_cds_num) % tb.size() == 0) ? ((c_amino_seq_len * c_cds_num) / tb.size()) : ((c_amino_seq_len * c_cds_num) / tb.size()) + 1;
    for (int i = 0; i < partition_num; i++)
    {
        idx = tb.size() * i + tb.thread_rank();
        amino_seq_idx = idx % c_amino_seq_len;
        aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
        solution_idx = idx * CODON_SIZE;
        if (idx < (c_amino_seq_len * c_cds_num))
        {
            gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
            if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
            {
                continue;
            }

            char cur_codon_idx;
            char new_codon_idx;

            cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);

            new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
            while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1))
            {
                new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
            }

            if (new_codon_idx == cur_codon_idx)
            {
                new_codon_idx += 1;
            }

            solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
            solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
            solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
        }
    }

    return;
}

/*
1. SELECT_UPPER : Selecting the highest weight codon
2. SELECT_UPPER_RANDOM : Selecting codon among having high weight codons than current codon weight
3. SELECT_RANDOM : Selecting Selecting random codon except current codon
*/
__device__ void mutationCAI(const thread_block tb, curandStateXORWOW *random_generator, char *solution, const char *s_amino_seq_idx, const char *s_obj_idx, const char mutation_type, const float mutation_prob = c_mutation_prob)
{
    int partition_num;
    int idx;
    int amino_seq_idx;
    int solution_idx;
    char aminoacid_idx;
    float gen_prob;

    partition_num = (c_amino_seq_len % tb.size() == 0) ? (c_amino_seq_len / tb.size()) : (c_amino_seq_len / tb.size()) + 1;
    for (int i = 0; i < partition_num; i++)
    {
        idx = tb.size() * i + tb.thread_rank();
        amino_seq_idx = idx;
        aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
        solution_idx = c_cds_len * s_obj_idx[MIN_CAI_IDX * 2] + idx * CODON_SIZE;
        if (idx < c_amino_seq_len)
        {
            gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
            if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
            {
                continue;
            }

            char cur_codon_idx;
            char new_codon_idx;

            cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);

            switch (mutation_type)
            {
            case SELECT_UPPER:
                if (cur_codon_idx != c_syn_codons_num[aminoacid_idx] - 1)
                {
                    solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE];
                    solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE + 1];
                    solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE + 2];
                }
                break;

            case SELECT_UPPER_RANDOM:
                if (cur_codon_idx != c_syn_codons_num[aminoacid_idx] - 1)
                {
                    new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1 - cur_codon_idx));
                    while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1 - cur_codon_idx))
                    {
                        new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1 - cur_codon_idx));
                    }
                    new_codon_idx += 1;

                    solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
                    solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
                    solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
                }
                break;

            case SELECT_RANDOM:
                new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
                while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1))
                {
                    new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
                }

                if (new_codon_idx == cur_codon_idx)
                {
                    new_codon_idx += 1;
                }

                solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
                solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
                solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
                break;
            }
        }
    }

    return;
}

/*
Even, Odd
1. SELECT_UPPER : Selecting the highest codon's CPS sum
2. SELECT_UPPER_RANDOM : Selecting codon among having high codons's CPS sum than current codon's CPS sum
3. SELECT_RANDOM : Selecting Selecting random codon except current codon
*/
__device__ void mutationCBP(const thread_block tb, curandStateXORWOW *random_generator, char *solution, const char *s_amino_seq_idx, const char *s_obj_idx, const char mutation_type, const char type_even_odd, const float mutation_prob = c_mutation_prob * 2)
{
    int partition_num;
    int idx;
    int amino_seq_idx;
    int solution_idx;
    char left_aminoacid_idx, aminoacid_idx, right_aminoacid_idx;
    float gen_prob;

    partition_num = ((c_amino_seq_len - 1) % tb.size() == 0) ? ((c_amino_seq_len - 1) / tb.size()) : ((c_amino_seq_len - 1) / tb.size()) + 1;
    for (int i = 0; i < partition_num; i++)
    {
        idx = tb.size() * i + tb.thread_rank();
        amino_seq_idx = idx;
        left_aminoacid_idx = s_amino_seq_idx[amino_seq_idx - 1];
        aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
        right_aminoacid_idx = s_amino_seq_idx[amino_seq_idx + 1];
        solution_idx = c_cds_len * s_obj_idx[MIN_CBP_IDX * 2] + idx * CODON_SIZE;
        if ((idx < (c_amino_seq_len - 1)) && ((idx % 2) == type_even_odd))
        {
            gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
            if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1) || (aminoacid_idx == 20))
            {
                continue;
            }

            char cur_codon_idx;
            char new_codon_idx;
            char left_codon_idx;
            char right_codon_idx;

            char upper_cps_sum_codons_idx[MAX_SYN_CODONS_NUM - 1];
            char upper_cps_sum_codons_cnt = 0;
            char max_cps_sum_codon_idx;

            float cur_cps_sum;
            float max_cps_sum;

            cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);
            left_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx - CODON_SIZE], &c_codons[c_codons_start_idx[left_aminoacid_idx] * CODON_SIZE], c_syn_codons_num[left_aminoacid_idx]);
            right_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx + CODON_SIZE], &c_codons[c_codons_start_idx[right_aminoacid_idx] * CODON_SIZE], c_syn_codons_num[right_aminoacid_idx]);

            cur_cps_sum = c_cps[(c_codons_start_idx[left_aminoacid_idx] + left_codon_idx) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[aminoacid_idx] + cur_codon_idx];
            if (right_codon_idx < (TOTAL_CODON_NUM - STOP_CODON_NUM))
            {
                cur_cps_sum += c_cps[(c_codons_start_idx[aminoacid_idx] + cur_codon_idx) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[right_aminoacid_idx] + right_codon_idx];
            }

            max_cps_sum = cur_cps_sum;
            for (char j = 0; j < c_syn_codons_num[aminoacid_idx]; j++)
            {
                float new_cps_sum = 0.f;

                if (j != cur_codon_idx)
                {
                    new_cps_sum += c_cps[(c_codons_start_idx[left_aminoacid_idx] + left_codon_idx) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[aminoacid_idx] + j];
                    if (right_codon_idx < (TOTAL_CODON_NUM - STOP_CODON_NUM))
                    {
                        new_cps_sum += c_cps[(c_codons_start_idx[aminoacid_idx] + j) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[right_aminoacid_idx] + right_codon_idx];
                    }

                    if (new_cps_sum > cur_cps_sum)
                    {
                        upper_cps_sum_codons_idx[upper_cps_sum_codons_cnt++] = j;
                        if (new_cps_sum > max_cps_sum)
                        {
                            max_cps_sum_codon_idx = j;
                        }
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
                {
                    new_codon_idx += 1;
                }

                solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
                solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
                solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
                break;
            }
        }
    }
    return;
}

/*
1. SELECT_UPPER : Selecting the highest HSC sum
2. SELECT_UPPER_RANDOM : Selecting codon among having high HSC sum than current HSC sum
3. SELECT_RANDOM : Selecting Selecting random codon except current codon
If there are multiple good codon, one codon is selected randomly
*/
__device__ void mutationHSC(const thread_block tb, curandStateXORWOW *random_generator, char *solution, const char *s_amino_seq_idx, const char *s_obj_idx, const char mutation_type, const char type_even_odd, const float mutation_prob = c_mutation_prob * 2)
{
    int partition_num;
    int idx;
    int amino_seq_idx;
    int solution_idx;
    char aminoacid_idx;
    float gen_prob;

    partition_num = (c_amino_seq_len % tb.size() == 0) ? (c_amino_seq_len / tb.size()) : (c_amino_seq_len / tb.size()) + 1;
    for (int i = 0; i < partition_num; i++)
    {
        idx = tb.size() * i + tb.thread_rank();
        amino_seq_idx = idx;
        aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
        solution_idx = c_cds_len * s_obj_idx[MIN_HSC_IDX * 2] + idx * CODON_SIZE;
        if ((idx < c_amino_seq_len) && ((idx % 2) == type_even_odd))
        {
            gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
            if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
            {
                continue;
            }

            char cur_codon_idx;
            char new_codon_idx;

            char upper_hsc_sum_codons_idx[MAX_SYN_CODONS_NUM - 1];
            char upper_hsc_sum_codons_cnt = 0;
            char max_hsc_sum_codon_idx;

            char cur_hsc_sum;
            char max_hsc_sum;

            bool check;

            check = isStopCodon(&solution[solution_idx]); // if current codon is stop codon, calculating right hsc is not needed

            char index_array[MAX_SYN_CODONS_NUM]; // For selecting randomly synonymous codons
            for (char j = 0; j < MAX_SYN_CODONS_NUM; j++)
            {
                if (j < c_syn_codons_num[aminoacid_idx])
                {
                    index_array[j] = j;
                }
                else
                {
                    index_array[j] = EMPTY;
                }
            }
            indexArrayShuffling(random_generator, index_array, c_syn_codons_num[aminoacid_idx]);

            cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);
            if (check)
            {
                cur_hsc_sum = countLeftSideHSC(&solution[solution_idx - CODON_SIZE], &solution[solution_idx]);
            }
            else
            {
                cur_hsc_sum = countBothSidesHSC(&solution[solution_idx - CODON_SIZE], &solution[solution_idx], &solution[solution_idx + CODON_SIZE]);
            }

            max_hsc_sum = cur_hsc_sum;
            for (char j = 0; j < c_syn_codons_num[aminoacid_idx]; j++)
            {
                char new_hsc_sum = 0;
                char shuffle_idx = index_array[j];

                if (shuffle_idx != cur_codon_idx)
                {
                    if (check)
                    {
                        new_hsc_sum = countLeftSideHSC(&solution[solution_idx - CODON_SIZE], &c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE]);
                    }
                    else
                    {
                        new_hsc_sum = countBothSidesHSC(&solution[solution_idx - CODON_SIZE], &c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE], &solution[solution_idx + CODON_SIZE]);
                    }

                    if (new_hsc_sum > cur_hsc_sum)
                    {
                        upper_hsc_sum_codons_idx[upper_hsc_sum_codons_cnt++] = shuffle_idx;
                        if (new_hsc_sum > max_hsc_sum)
                        {
                            max_hsc_sum_codon_idx = shuffle_idx;
                        }
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
                {
                    new_codon_idx += 1;
                }

                solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
                solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
                solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
                break;
            }
        }
    }
    return;
}

/*
This is same as MOBOA & MaOMPE
codon_idx is current codon start point assumming that CDS is one
*/
__device__ void mutationHD(const thread_block tb, curandStateXORWOW *random_generator, char *solution, const char *s_amino_seq_idx, const char *s_obj_idx, const float mutation_prob = c_mutation_prob)
{
    int partition_num;
    int idx;
    int amino_seq_idx;
    char aminoacid_idx;
    char cds1_idx, cds2_idx;
    int codon_idx;
    float gen_prob;

    partition_num = (c_amino_seq_len % tb.size() == 0) ? (c_amino_seq_len / tb.size()) : (c_amino_seq_len / tb.size()) + 1;
    for (int i = 0; i < partition_num; i++)
    {
        idx = tb.size() * i + tb.thread_rank();
        amino_seq_idx = idx;
        aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
        cds1_idx = s_obj_idx[MIN_HD_IDX * 2];
        cds2_idx = s_obj_idx[MIN_HD_IDX * 2 + 1];
        codon_idx = amino_seq_idx * CODON_SIZE;
        if (idx < c_amino_seq_len)
        {

            gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
            if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
            {
                continue;
            }

            char cur_HD_pairCDSs;
            char cur_mHD;
            char best_HD_pairCDSs;
            char best_mHD;
            char new_HD_pairCDSs;
            char new_mHD;

            char cur_codon_idx;
            char best_syn_pos;
            char tmp;

            char index_array[MAX_SYN_CODONS_NUM];
            for (char j = 0; j < MAX_SYN_CODONS_NUM; j++)
            {
                if (j < c_syn_codons_num[aminoacid_idx])
                {
                    index_array[j] = j;
                }
                else
                {
                    index_array[j] = EMPTY;
                }
            }
            indexArrayShuffling(random_generator, index_array, c_syn_codons_num[aminoacid_idx]);

            cur_codon_idx = findIndexAmongSynonymousCodons(&solution[c_cds_len * cds1_idx + codon_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);

            cur_HD_pairCDSs = 0;
            for (int j = 0; j < CODON_SIZE; j++)
            {
                if (solution[c_cds_len * cds1_idx + codon_idx + j] != solution[c_cds_len * cds2_idx + codon_idx + j])
                {
                    cur_HD_pairCDSs += 1;
                }
            }
            cur_mHD = 127;

            for (int j = 0; j < c_cds_num; j++)
            {
                tmp = 0;
                if (j != cds1_idx)
                {
                    for (int k = 0; k < CODON_SIZE; k++)
                    {
                        if (solution[c_cds_len * cds1_idx + codon_idx + k] != solution[c_cds_len * j + codon_idx + k])
                        {
                            tmp += 1;
                        }
                    }

                    if (tmp < cur_mHD)
                    {
                        cur_mHD = tmp;
                    }
                }
            }

            best_HD_pairCDSs = -1;
            best_mHD = -1;
            for (int j = 0; j < c_syn_codons_num[aminoacid_idx]; j++)
            {
                char shuffle_idx = index_array[j];
                if (shuffle_idx != cur_codon_idx)
                {
                    new_HD_pairCDSs = 0;
                    for (int k = 0; k < CODON_SIZE; k++)
                    {
                        if (c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE + k] != solution[c_cds_len * cds2_idx + codon_idx + k])
                        {
                            new_HD_pairCDSs += 1;
                        }
                    }

                    new_mHD = 127;
                    for (int k = 0; k < c_cds_num; k++)
                    {
                        tmp = 0;
                        if (k != cds1_idx)
                        {
                            for (int l = 0; l < CODON_SIZE; l++)
                            {
                                if (c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE + l] != solution[c_cds_len * k + codon_idx + l])
                                {
                                    tmp += 1;
                                }
                            }

                            if (tmp < new_mHD)
                            {
                                new_mHD = tmp;
                            }
                        }
                    }

                    if ((new_mHD > cur_mHD) && (new_mHD > best_mHD))
                    {
                        best_mHD = new_mHD;
                        best_syn_pos = shuffle_idx;
                    }
                    else if ((best_mHD == -1) && (new_mHD == cur_mHD) && (new_HD_pairCDSs > cur_HD_pairCDSs) && (new_HD_pairCDSs > best_HD_pairCDSs))
                    {
                        best_HD_pairCDSs = new_HD_pairCDSs;
                        best_syn_pos = shuffle_idx;
                    }
                }
            }

            if (best_mHD != -1 || best_HD_pairCDSs != -1)
            {
                solution[cds1_idx * c_cds_len + codon_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + best_syn_pos) * CODON_SIZE];
                solution[cds1_idx * c_cds_len + codon_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + best_syn_pos) * CODON_SIZE + 1];
                solution[cds1_idx * c_cds_len + codon_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + best_syn_pos) * CODON_SIZE + 2];
            }
        }
    }
    return;
}

/*
1. SELECT_HIGH_GC : Selecting randomly high GC contents than current codon
2. SELECT_LOW_GC : Selecting randomly low GC contents than current codon
3. SELECT_RANDOM : Selecting Selecting random codon except current codon
input mutationp prob is computed before call this function
*/
__device__ void mutationGC(const thread_block tb, curandStateXORWOW *random_generator, char *solution, const char *s_amino_seq_idx, const char *s_obj_idx, const char mutation_type, const float mutation_prob = c_mutation_prob)
{
    int partition_num;
    int idx;
    int amino_seq_idx;
    int solution_idx;
    char aminoacid_idx;

    partition_num = (c_amino_seq_len % tb.size() == 0) ? (c_amino_seq_len / tb.size()) : (c_amino_seq_len / tb.size()) + 1;
    for (int i = 0; i < partition_num; i++)
    {
        idx = tb.size() * i + tb.thread_rank();
        amino_seq_idx = idx;
        aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
        solution_idx = c_cds_len * s_obj_idx[MAX_GC_IDX * 2] + idx * CODON_SIZE;
        if (idx < c_amino_seq_len)
        {
            float gen_prob;
            gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
            if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
            {
                continue;
            }

            char cur_codon_idx;
            char shuffle_idx;

            char cur_gc_sum;
            char new_gc_sum;

            bool check;

            check = false;

            char index_array[MAX_SYN_CODONS_NUM];
            for (char j = 0; j < MAX_SYN_CODONS_NUM; j++)
            {
                if (j < c_syn_codons_num[aminoacid_idx])
                {
                    index_array[j] = j;
                }
                else
                {
                    index_array[j] = EMPTY;
                }
            }
            indexArrayShuffling(random_generator, index_array, c_syn_codons_num[aminoacid_idx]);

            cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);
            cur_gc_sum = countCodonGC(&solution[solution_idx]);
            switch (mutation_type)
            {
            case SELECT_HIGH_GC:
                for (char j = 0; j < c_syn_codons_num[aminoacid_idx]; j++)
                {
                    shuffle_idx = index_array[j];
                    if (shuffle_idx != cur_codon_idx)
                    {
                        new_gc_sum = countCodonGC(&c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE]);
                        if (new_gc_sum > cur_gc_sum)
                        {
                            check = true;
                            break;
                        }
                    }
                }

                if (check)
                {
                    solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE];
                    solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE + 1];
                    solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE + 2];
                }
                break;

            case SELECT_LOW_GC:
                for (char j = 0; j < c_syn_codons_num[aminoacid_idx]; j++)
                {
                    shuffle_idx = index_array[j];
                    if (shuffle_idx != cur_codon_idx)
                    {
                        new_gc_sum = countCodonGC(&c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE]);
                        if (new_gc_sum < cur_gc_sum)
                        {
                            check = true;
                            break;
                        }
                    }
                }

                if (check)
                {
                    solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE];
                    solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE + 1];
                    solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE + 2];
                }
                break;

            case SELECT_RANDOM:
                char new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
                while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1))
                {
                    new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1));
                }

                if (new_codon_idx == cur_codon_idx)
                {
                    new_codon_idx += 1;
                }

                solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE];
                solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 1];
                solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + new_codon_idx) * CODON_SIZE + 2];
                break;
            }
        }
    }
    return;
}

/*
SL 을 깨기위한 변이 방법은 조금 고민해 볼 부분
*/
__device__ void mutationSL(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx)
{
    

    return;
}

#endif
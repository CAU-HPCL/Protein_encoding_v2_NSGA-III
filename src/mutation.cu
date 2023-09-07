#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../include/common.cuh"
#include "../include/utils.cuh"

#define SELECT_UPPER 0
#define SELECT_UPPER_RANDOM 1
#define SELECT_RANDOM 2
#define SELECT_HIGH_GC 3
#define SELECT_LOW_GC 4

__device__ void mutationRandom(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx)
{
    /*
    solution_idx : solution 에서 현재 쓰레드가 보는 코돈의 시작 인덱스 값
    aminoacid_idx : 현재 쓰레드가 보는 코돈의 아미노산 인덱스 값
    */

    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
    {
        return;
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

    return;
}

__device__ void mutationCAI(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx, const char mutation_type)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
    {
        return;
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

    return;
}

__device__ void mutationCBP(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char left_aminoacid_idx, const char aminoacid_idx, const char right_aminoacid_idx, const char mutation_type)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1) || (aminoacid_idx == 20))
    {
        return;
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
    for (char i = 0; i < c_syn_codons_num[aminoacid_idx]; i++)
    {
        float new_cps_sum = 0.f;

        if (i != cur_codon_idx)
        {
            new_cps_sum += c_cps[(c_codons_start_idx[left_aminoacid_idx] + left_codon_idx) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[aminoacid_idx] + i];
            if (right_codon_idx < (TOTAL_CODON_NUM - STOP_CODON_NUM))
            {
                cur_cps_sum += c_cps[(c_codons_start_idx[aminoacid_idx] + i) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + c_codons_start_idx[right_aminoacid_idx] + right_codon_idx];
            }

            if (new_cps_sum > cur_cps_sum)
            {
                upper_cps_sum_codons_idx[upper_cps_sum_codons_cnt++] = i;
                if (new_cps_sum > max_cps_sum)
                {
                    max_cps_sum_codon_idx = i;
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

    return;
}

__device__ void mutationHSC(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx, const char mutation_type)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
    {
        return;
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
    for (char i = 0; i < MAX_SYN_CODONS_NUM; i++)
    {
        if (i < c_syn_codons_num[aminoacid_idx])
        {
            index_array[i] = i;
        }
        else
        {
            index_array[i] = EMPTY;
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
    for (char i = 0; i < c_syn_codons_num[aminoacid_idx]; i++)
    {
        char new_hsc_sum = 0;
        char shuffle_idx = index_array[i];

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

    return;
}

__device__ void mutationHD(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int codon_idx, const char aminoacid_idx, const char cds1_idx, const char cds2_idx)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
    {
        return;
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
    for (char i = 0; MAX_SYN_CODONS_NUM; i++)
    {
        if (i < c_syn_codons_num[aminoacid_idx])
        {
            index_array[i] = i;
        }
        else
        {
            index_array[i] = EMPTY;
        }
    }
    indexArrayShuffling(random_generator, index_array, c_syn_codons_num[aminoacid_idx]);

    cur_codon_idx = findIndexAmongSynonymousCodons(&solution[c_cds_len * cds1_idx + codon_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);

    cur_HD_pairCDSs = 0;
    for (int i = 0; i < CODON_SIZE; i++)
    {
        if (solution[c_cds_len * cds1_idx + codon_idx + i] != solution[c_cds_len * cds2_idx + codon_idx + i])
        {
            cur_HD_pairCDSs += 1;
        }
    }
    cur_mHD = 127;

    for (int i = 0; i < c_cds_num; i++)
    {
        tmp = 0;
        if (i != cds1_idx)
        {
            for (int j = 0; j < CODON_SIZE; j++)
            {
                if (solution[c_cds_len * cds1_idx + codon_idx + j] != solution[c_cds_len * i + codon_idx + j])
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
    for (int i = 0; i < c_syn_codons_num[aminoacid_idx]; i++)
    {
        char shuffle_idx = index_array[i];
        if (shuffle_idx != cur_codon_idx)
        {
            new_HD_pairCDSs = 0;
            for (int j = 0; j < CODON_SIZE; j++)
            {
                if (c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE + j] != solution[c_cds_len * cds2_idx + codon_idx + j])
                {
                    new_HD_pairCDSs += 1;
                }
            }

            new_mHD = 127;
            for (int j = 0; j < c_cds_num; j++)
            {
                tmp = 0;
                if (j != cds1_idx)
                {
                    for (int k = 0; k < CODON_SIZE; k++)
                    {
                        if (c_codons[(c_codons_start_idx[aminoacid_idx] + shuffle_idx) * CODON_SIZE + k] != solution[c_cds_len * j + codon_idx + k])
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

    return;
}

__device__ void mutationGC(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const char solution_idx, const char aminoacid_idx, const char mutation_type)
{
    float gen_prob;
    gen_prob = curand_uniform(random_generator); // 1.0 is included and 0.0 is excluded
    if ((gen_prob > mutation_prob) || (c_syn_codons_num[aminoacid_idx] == 1))
    {
        return;
    }

    char cur_codon_idx;
    char shuffle_idx;

    char cur_gc_sum;
    char new_gc_sum;

    bool check;

    check = false;

    char index_array[MAX_SYN_CODONS_NUM];
    for (char i = 0; MAX_SYN_CODONS_NUM; i++)
    {
        if (i < c_syn_codons_num[aminoacid_idx])
        {
            index_array[i] = i;
        }
        else
        {
            index_array[i] = EMPTY;
        }
    }
    indexArrayShuffling(random_generator, index_array, c_syn_codons_num[aminoacid_idx]);

    cur_codon_idx = findIndexAmongSynonymousCodons(&solution[solution_idx], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);

    cur_gc_sum = countCodonGC(&c_codons[(c_codons_start_idx[aminoacid_idx] + cur_codon_idx) * CODON_SIZE]);
    switch (mutation_type)
    {
    case SELECT_HIGH_GC:
        for (char i = 0; i < c_syn_codons_num[aminoacid_idx]; i++)
        {
            shuffle_idx = index_array[i];
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
        for (char i = 0; i < c_syn_codons_num[aminoacid_idx]; i++)
        {
            shuffle_idx = index_array[i];
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

    return;
}


/*
SL 을 깨기위한 변이 방법은 조금 고민해 볼 부분
*/
__device__ void mutationSL(const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const char aminoacid_idx)
{

    return;
}
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>
#include <cooperative_groups.h>

#include "../include/common.cuh"
#include "../include/info.cuh"

using namespace cooperative_groups;

__host__ char findAminoIndex(const char amino_abbreviation)
{
    char low = 0;
    char high = 21 - 1;
    char mid;

    while (low <= high)
    {
        mid = (low + high) / 2;

        if (aminoacids[mid] == amino_abbreviation)
        {
            return mid;
        }
        else if (aminoacids[mid] > amino_abbreviation)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }

    printf("FindAminoIndex Function failure\n");
    exit(EXIT_FAILURE);
}

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
    printf("%s\n",cur_codon);
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

__device__ char countRightSideHSC(const char *cur_codon, const char *right_codon)
{
    char result = 0;

    char temp_codon[3];

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

__device__ bool isCompatible(const char base1, const char base2)
{
    if ((base1 == 'A') && (base2 == 'U'))
    {
        return true;
    }
    else if ((base1 == 'U') && (base2 == 'A'))
    {
        return true;
    }
    else if ((base1 == 'G') && (base2 == 'C'))
    {
        return true;
    }
    else if ((base1 == 'C') && (base2 == 'G'))
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ void calMinimumCAI(const thread_block tb, const char *solution, const char *s_amino_seq_idx, float *s_obj_buffer, float *s_obj_val, char *s_obj_idx)
{
    int partition_num;

    int i, j;
    int amino_seq_idx;
    char aminoacid_idx;
    char cur_codon_idx;

    partition_num = (c_amino_seq_len % tb.size() == 0) ? (c_amino_seq_len / tb.size()) : (c_amino_seq_len / tb.size()) + 1;
    for (i = 0; i < c_cds_num; i++)
    {
        s_obj_buffer[tb.thread_rank()] = 1.f;

        for (j = 0; j < partition_num; j++)
        {
            amino_seq_idx = tb.size() * j + tb.thread_rank();
            aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
            if (amino_seq_idx < c_amino_seq_len)
            {
                cur_codon_idx = findIndexAmongSynonymousCodons(&solution[c_cds_len * i + amino_seq_idx * CODON_SIZE], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);
                s_obj_buffer[tb.thread_rank()] *= (float)pow(c_codons_weight[c_codons_start_idx[aminoacid_idx] + cur_codon_idx], 1.0 / c_amino_seq_len);
            }
        }
        tb.sync();

        j = tb.size() / 2;
        while (true)
        {
            if (tb.thread_rank() < j)
            {
                s_obj_buffer[tb.thread_rank()] *= s_obj_buffer[tb.thread_rank() + j];
            }
            tb.sync();

            if (j == 1)
            {
                break;
            }

            if ((j % 2 == 1) && (tb.thread_rank() == 0))
            {
                s_obj_buffer[0] *= s_obj_buffer[j - 1];
            }
            tb.sync();

            j /= 2;
        }

        if (tb.thread_rank() == 0)
        {
            if (i == 0)
            {
                s_obj_val[MIN_CAI_IDX] = s_obj_buffer[0];
                s_obj_idx[MIN_CAI_IDX * 2] = i;
            }
            else if (s_obj_buffer[0] <= s_obj_val[MIN_CAI_IDX])
            {
                s_obj_val[MIN_CAI_IDX] = s_obj_buffer[0];
                s_obj_idx[MIN_CAI_IDX * 2] = i;
            }
        }
        tb.sync();
    }

    return;
}

__device__ void calMinimumCBP(const thread_block tb, const char *solution, const char *s_amino_seq_idx, float *s_obj_buffer, float *s_obj_val, char *s_obj_idx)
{
    int partition_num;

    int i, j;
    int amino_seq_idx;
    char aminoacid_idx;
    char right_aminoacid_idx;
    char cur_codon_idx;
    char right_codon_idx;

    partition_num = ((c_amino_seq_len - 2) % tb.size() == 0) ? ((c_amino_seq_len - 2) / tb.size()) : ((c_amino_seq_len - 2) / tb.size()) + 1;
    for (i = 0; i < c_cds_num; i++)
    {
        s_obj_buffer[tb.thread_rank()] = 0.f;

        for (j = 0; j < partition_num; j++)
        {
            amino_seq_idx = tb.size() * j + tb.thread_rank();
            aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
            right_aminoacid_idx = s_amino_seq_idx[amino_seq_idx + 1];
            if (amino_seq_idx < (c_amino_seq_len - 2))
            {
                cur_codon_idx = findIndexAmongSynonymousCodons(&solution[c_cds_len * i + amino_seq_idx * CODON_SIZE], &c_codons[c_codons_start_idx[aminoacid_idx] * CODON_SIZE], c_syn_codons_num[aminoacid_idx]);
                right_codon_idx = findIndexAmongSynonymousCodons(&solution[c_cds_len * i + (amino_seq_idx + 1) * CODON_SIZE], &c_codons[c_codons_start_idx[right_aminoacid_idx] * CODON_SIZE], c_syn_codons_num[right_aminoacid_idx]);
                s_obj_buffer[tb.thread_rank()] += c_cps[(c_codons_start_idx[aminoacid_idx] + cur_codon_idx) * (TOTAL_CODON_NUM - STOP_CODON_NUM) + (c_codons_start_idx[right_aminoacid_idx] + right_codon_idx)];
            }
        }
        tb.sync();

        j = tb.size() / 2;
        while (true)
        {
            if (tb.thread_rank() < j)
            {
                s_obj_buffer[tb.thread_rank()] += s_obj_buffer[tb.thread_rank() + j];
            }
            tb.sync();

            if (j == 1)
            {
                break;
            }

            if ((j % 2 == 1) && (tb.thread_rank() == 0))
            {
                s_obj_buffer[0] += s_obj_buffer[j - 1];
            }
            tb.sync();

            j /= 2;
        }

        if (tb.thread_rank() == 0)
        {
            if (i == 0)
            {
                s_obj_val[MIN_CBP_IDX] = s_obj_buffer[0] / (c_amino_seq_len - 2);
                s_obj_idx[MIN_CBP_IDX * 2] = i;
            }
            else if ((s_obj_buffer[0] / (c_amino_seq_len - 2)) <= s_obj_val[MIN_CBP_IDX])
            {
                s_obj_val[MIN_CBP_IDX] = s_obj_buffer[0] / (c_amino_seq_len - 2);
                s_obj_idx[MIN_CBP_IDX * 2] = i;
            }
        }
        tb.sync();
    }

    return;
}

__device__ void calMinimumHSC(const thread_block tb, const char *solution, const char *s_amino_seq_idx, float *s_obj_buffer, float *s_obj_val, char *s_obj_idx)
{
    int partition_num;

    int i, j;
    int amino_seq_idx;

    partition_num = ((c_amino_seq_len - 1) % tb.size() == 0) ? ((c_amino_seq_len - 1) / tb.size()) : ((c_amino_seq_len - 1) / tb.size()) + 1;
    for (i = 0; i < c_cds_num; i++)
    {
        s_obj_buffer[tb.thread_rank()] = 0.f;

        for (j = 0; j < partition_num; j++)
        {
            amino_seq_idx = tb.size() * j + tb.thread_rank();
            if (amino_seq_idx < (c_amino_seq_len - 1))
            {
                s_obj_buffer[tb.thread_rank()] += countRightSideHSC(&solution[c_cds_len * i + amino_seq_idx * CODON_SIZE], &solution[c_cds_len * i + (amino_seq_idx + 1) * CODON_SIZE]);
            }
        }
        tb.sync();

        j = tb.size() / 2;
        while (true)
        {
            if (tb.thread_rank() < j)
            {
                s_obj_buffer[tb.thread_rank()] += s_obj_buffer[tb.thread_rank() + j];
            }
            tb.sync();

            if (j == 1)
            {
                break;
            }

            if ((j % 2 == 1) && (tb.thread_rank() == 0))
            {
                s_obj_buffer[0] += s_obj_buffer[j - 1];
            }
            tb.sync();

            j /= 2;
        }

        if (tb.thread_rank() == 0)
        {
            if (i == 0)
            {
                s_obj_val[MIN_HSC_IDX] = s_obj_buffer[0];
                s_obj_idx[MIN_HSC_IDX * 2] = i;
            }
            else if (s_obj_buffer[0] <= s_obj_val[MIN_HSC_IDX])
            {
                s_obj_val[MIN_HSC_IDX] = s_obj_buffer[0];
                s_obj_idx[MIN_HSC_IDX * 2] = i;
            }
        }
        tb.sync();
    }

    return;
}

__device__ void calMinimumHD(const thread_block tb, const char *solution, const char *s_amino_seq_idx, float *s_obj_buffer, float *s_obj_val, char *s_obj_idx)
{
    int partition_num;

    int i, j, k;
    int base_idx;

    partition_num = (c_cds_len % tb.size() == 0) ? (c_cds_len / tb.size()) : (c_cds_len / tb.size()) + 1;
    for (i = 0; i < c_cds_num - 1; i++)
    {
        for (j = i + 1; j < c_cds_num; j++)
        {
            s_obj_buffer[tb.thread_rank()] = 0.f;

            for (k = 0; k < partition_num; k++)
            {
                base_idx = tb.size() * k + tb.thread_rank();

                if ((base_idx < c_cds_len) && (solution[c_cds_len * i + base_idx] != solution[c_cds_len * j + base_idx]))
                {
                    s_obj_buffer[tb.thread_rank()] += 1;
                }
            }
            tb.sync();

            k = tb.size() / 2;
            while (true)
            {
                if (tb.thread_rank() < k)
                {
                    s_obj_buffer[tb.thread_rank()] += s_obj_buffer[tb.thread_rank() + k];
                }
                tb.sync();

                if (k == 1)
                {
                    break;
                }

                if ((k % 2 == 1) && (tb.thread_rank() == 0))
                {
                    s_obj_buffer[0] += s_obj_buffer[k - 1];
                }
                tb.sync();

                k /= 2;
            }

            if (tb.thread_rank() == 0)
            {
                if (i == 0 && j == 1)
                {
                    s_obj_val[MIN_HD_IDX] = s_obj_buffer[0];
                    s_obj_idx[MIN_HD_IDX * 2] = i;
                    s_obj_idx[MIN_HD_IDX * 2 + 1] = j;
                }
                else if (s_obj_buffer[0] <= s_obj_val[MIN_HD_IDX])
                {
                    s_obj_val[MIN_HD_IDX] = s_obj_buffer[0];
                    s_obj_idx[MIN_HD_IDX * 2] = i;
                    s_obj_idx[MIN_HD_IDX * 2 + 1] = j;
                }
            }
            tb.sync();
        }
    }

    return;
}

__device__ void calMaximumGC(const thread_block tb, const char *solution, const char *s_amino_seq_idx, float *s_obj_buffer, float *s_obj_val, char *s_obj_idx)
{
    int partition_num;

    int i, j;
    int base_idx;

    partition_num = (c_cds_len % tb.size() == 0) ? (c_cds_len / tb.size()) : (c_cds_len / tb.size()) + 1;
    for (i = 0; i < c_cds_num; i++)
    {
        s_obj_buffer[tb.thread_rank()] = 0.f;

        for (j = 0; j < partition_num; j++)
        {
            base_idx = tb.size() * j + tb.thread_rank();

            if (base_idx < c_cds_len)
            {
                if (solution[c_cds_len * i + base_idx] == 'G' || solution[c_cds_len * i + base_idx] == 'C')
                {
                    s_obj_buffer[tb.thread_rank()] += 1;
                }
            }
        }
        tb.sync();

        j = tb.size() / 2;
        while (true)
        {
            if (tb.thread_rank() < j)
            {
                s_obj_buffer[tb.thread_rank()] += s_obj_buffer[tb.thread_rank() + j];
            }
            tb.sync();

            if (j == 1)
            {
                break;
            }

            if ((j % 2 == 1) && (tb.thread_rank() == 0))
            {
                s_obj_buffer[0] += s_obj_buffer[j - 1];
            }
            tb.sync();

            j /= 2;
        }

        /*
        TODO : ideal GC 와 편차를 구하는 것이기 때문에 추가적 코딩 및 수정필요한 부분
        */
        if (tb.thread_rank() == 0)
        {
            if (i == 0)
            {
                s_obj_val[MAX_GC_IDX] = s_obj_buffer[0];
                s_obj_idx[MAX_GC_IDX * 2] = i;
            }
            else if (s_obj_buffer[0] >= s_obj_val[MAX_GC_IDX])
            {
                s_obj_val[MAX_GC_IDX] = s_obj_buffer[0];
                s_obj_idx[MAX_GC_IDX * 2] = i;
            }
        }
        tb.sync();
    }

    return;
}

__device__ void calMaximumSL(const thread_block tb, const char *solution, const char *s_amino_seq_idx, float *s_obj_buffer, float *s_obj_val, char *s_obj_idx, int *s_pql, int *s_mutex)
{
    int p, q, l;
    char cds_idx;
    int st_r, st_c;

    int i, j;
    int tmp_l;
    int t_idx, diag_len;

    s_obj_buffer[tb.thread_rank()] = EMPTY;
    l = 0;
    for (i = 0; i < c_cds_num; i++)
    {
        t_idx = tb.thread_rank();
        while (t_idx < 2 * c_cds_len + 1)
        {
            if (t_idx < c_cds_len + 1)
            {
                diag_len = t_idx + 1;
                st_r = c_cds_len - diag_len;
                st_c = c_cds_len;
                for (j = 0; j < diag_len; j++)
                {
                    if (j == 0)
                    {
                        tmp_l = 0;
                    }
                    else if (isCompatible(solution[c_cds_len * i + st_r + j], solution[c_cds_len * i + st_c - j]))
                    {
                        tmp_l++;
                        if ((st_c - j) > (st_r + j))
                        {
                            if (((st_c - j) - (st_r + j)) > MIN_HAIRPIN_DISTANCE)
                            {
                                if (tmp_l >= l)
                                {
                                    l = tmp_l;
                                    s_obj_buffer[tb.thread_rank()] = l;
                                    p = st_r + j - l + 1;
                                    q = st_c - j;
                                    cds_idx = (char)i;
                                }
                            }
                        }
                        else if ((st_r + j - tmp_l + 1) > (st_c - j + tmp_l - 1))
                        {
                            if (((st_r + j - tmp_l + 1) - (st_c - j + tmp_l - 1)) > MIN_HAIRPIN_DISTANCE)
                            {
                                if (tmp_l >= l)
                                {
                                    l = tmp_l;
                                    s_obj_buffer[tb.thread_rank()] = l;
                                    p = st_c - j;
                                    q = st_r + j - l + 1;
                                    cds_idx = (char)i;
                                }
                            }
                        }
                        else
                        {
                            tmp_l = 0;
                        }
                    }
                    else
                    {
                        tmp_l = 0;
                    }
                }
            }
            else
            {
                diag_len = 2 * c_cds_len + 1 - t_idx;
                st_r = -1;
                st_c = diag_len - 1;
                for (j = 0; j < diag_len; j++)
                {
                    if (j == 0)
                    {
                        tmp_l = 0;
                    }
                    else if (isCompatible(solution[c_cds_len * i + st_r + j], solution[c_cds_len * i + st_c - j]))
                    {
                        tmp_l++;
                        if ((st_c - j) > (st_r + j))
                        {
                            if (((st_c - j) - (st_r + j)) > MIN_HAIRPIN_DISTANCE)
                            {
                                if (tmp_l >= l)
                                {
                                    l = tmp_l;
                                    s_obj_buffer[tb.thread_rank()] = l;
                                    p = st_r + j - l + 1;
                                    q = st_c - j;
                                    cds_idx = (char)i;
                                }
                            }
                        }
                        else if ((st_r + j - tmp_l + 1) > (st_c - j + tmp_l - 1))
                        {
                            if (((st_r + j - tmp_l + 1) - (st_c - j + tmp_l - 1)) > MIN_HAIRPIN_DISTANCE)
                            {
                                if (tmp_l >= l)
                                {
                                    l = tmp_l;
                                    s_obj_buffer[tb.thread_rank()] = l;
                                    p = st_c - j;
                                    q = st_r + j - l + 1;
                                    cds_idx = (char)i;
                                }
                            }
                        }
                        else
                        {
                            tmp_l = 0;
                        }
                    }
                    else
                    {
                        tmp_l = 0;
                    }
                }
            }
            t_idx += tb.size();
        }
    }
    tb.sync();

    i = tb.size() / 2;
    tb.sync();
    while (true)
    {
        if ((tb.thread_rank() < i) && (s_obj_buffer[tb.thread_rank() + i] > s_obj_buffer[tb.thread_rank()]))
        {
            s_obj_buffer[tb.thread_rank()] = s_obj_buffer[tb.thread_rank() + i];
        }
        tb.sync();

        if (i == 1)
        {
            break;
        }
        if ((i % 2 == 1) && (tb.thread_rank() == 0))
        {
            if (s_obj_buffer[i - 1] > s_obj_buffer[0])
            {
                s_obj_buffer[0] = s_obj_buffer[i - 1];
            }
        }
        tb.sync();

        i /= 2;
    }

    if (tb.thread_rank() == 0)
    {
        s_mutex[0] = 0;
    }
    tb.sync();
    if (l == s_obj_buffer[0])
    {
        while (atomicCAS(&s_mutex[0], 0, 1) != 0) // spin lock
        {
        }

        s_pql[P] = p;
        s_pql[Q] = q;
        s_pql[L] = l;
        s_obj_val[MAX_SL_IDX] = l;
        s_obj_idx[MAX_SL_IDX * 2] = cds_idx;

        atomicExch(&s_mutex[0], 0);
    }
    tb.sync();

    return;
}

__device__ void calOneCDS_SL(const thread_block tb, const char *solution, const char *s_amino_seq_idx, float *s_obj_buffer, char cds_idx, int *s_pql, int *s_mutex)
{
    int l;
    int st_r, st_c;

    int i;
    int tmp_l;
    int t_idx, diag_len;

    s_obj_buffer[tb.thread_rank()] = EMPTY;
    l = 0;
    t_idx = tb.thread_rank();
    while (t_idx < 2 * c_cds_len + 1)
    {
        if (t_idx < c_cds_len + 1)
        {
            diag_len = t_idx + 1;
            st_r = c_cds_len - diag_len;
            st_c = c_cds_len;
            for (i = 0; i < diag_len; i++)
            {
                if (i == 0)
                {
                    tmp_l = 0;
                }
                else if (isCompatible(solution[c_cds_len * cds_idx + st_r + i], solution[c_cds_len * cds_idx + st_c - i]))
                {
                    tmp_l++;
                    if ((st_c - i) > (st_r + i))
                    {
                        if (((st_c - i) - (st_r + i)) > MIN_HAIRPIN_DISTANCE)
                        {
                            if (tmp_l >= l)
                            {
                                l = tmp_l;
                                s_obj_buffer[tb.thread_rank()] = l;
                            }
                        }
                    }
                    else if ((st_r + i - tmp_l + 1) > (st_c - i + tmp_l - 1))
                    {
                        if (((st_r + i - tmp_l + 1) - (st_c - i + tmp_l - 1)) > MIN_HAIRPIN_DISTANCE)
                        {
                            if (tmp_l >= l)
                            {
                                l = tmp_l;
                                s_obj_buffer[tb.thread_rank()] = l;
                            }
                        }
                    }
                    else
                    {
                        tmp_l = 0;
                    }
                }
                else
                {
                    tmp_l = 0;
                }
            }
        }
        else
        {
            diag_len = 2 * c_cds_len + 1 - t_idx;
            st_r = -1;
            st_c = diag_len - 1;
            for (i = 0; i < diag_len; i++)
            {
                if (i == 0)
                {
                    tmp_l = 0;
                }
                else if (isCompatible(solution[c_cds_len * cds_idx + st_r + i], solution[c_cds_len * cds_idx + st_c - i]))
                {
                    tmp_l++;
                    if ((st_c - i) > (st_r + i))
                    {
                        if (((st_c - i) - (st_r + i)) > MIN_HAIRPIN_DISTANCE)
                        {
                            if (tmp_l >= l)
                            {
                                l = tmp_l;
                                s_obj_buffer[tb.thread_rank()] = l;
                            }
                        }
                    }
                    else if ((st_r + i - tmp_l + 1) > (st_c - i + tmp_l - 1))
                    {
                        if (((st_r + i - tmp_l + 1) - (st_c - i + tmp_l - 1)) > MIN_HAIRPIN_DISTANCE)
                        {
                            if (tmp_l >= l)
                            {
                                l = tmp_l;
                                s_obj_buffer[tb.thread_rank()] = l;
                            }
                        }
                    }
                    else
                    {
                        tmp_l = 0;
                    }
                }
                else
                {
                    tmp_l = 0;
                }
            }
        }
        t_idx += tb.size();
    }
    tb.sync();

    i = tb.size() / 2;
    tb.sync();
    while (true)
    {
        if ((tb.thread_rank() < i) && (s_obj_buffer[tb.thread_rank() + i] > s_obj_buffer[tb.thread_rank()]))
        {
            s_obj_buffer[tb.thread_rank()] = s_obj_buffer[tb.thread_rank() + i];
        }
        tb.sync();

        if (i == 1)
        {
            break;
        }
        if ((i % 2 == 1) && (tb.thread_rank() == 0))
        {
            if (s_obj_buffer[i - 1] > s_obj_buffer[0])
            {
                s_obj_buffer[0] = s_obj_buffer[i - 1];
            }
        }
        tb.sync();

        i /= 2;
    }

    if (tb.thread_rank() == 0)
    {
        s_mutex[0] = 0;
    }
    tb.sync();
    if (l == s_obj_buffer[0])
    {
        while (atomicCAS(&s_mutex[0], 0, 1) != 0) // spin lock
        {
        }
        s_pql[L] = l;

        atomicExch(&s_mutex[0], 0);
    }
    tb.sync();

    return;
}

__device__ void genPopulation(const thread_block tb, curandStateXORWOW *random_generator, const char *s_amino_seq_idx, char *s_solution, const char gen_type)
{
    int partition_num;

    int i, j, k;
    int idx;
    int amino_seq_idx;
    char pos;

    partition_num = ((c_amino_seq_len * c_cds_num) % tb.size() == 0) ? (c_amino_seq_len * c_cds_num) / tb.size() : (c_amino_seq_len * c_cds_num) / tb.size() + 1;
    switch (gen_type)
    {
    case RANDOM_GEN:
        for (i = 0; i < partition_num; i++)
        {
            idx = tb.size() * i + tb.thread_rank();
            if (idx < c_amino_seq_len * c_cds_num)
            {
                amino_seq_idx = idx % c_amino_seq_len;

                do
                {
                    pos = (char)(curand_uniform(random_generator) * c_syn_codons_num[s_amino_seq_idx[amino_seq_idx]]);
                } while (pos == c_syn_codons_num[s_amino_seq_idx[amino_seq_idx]]);

                j = idx * CODON_SIZE;
                k = (c_codons_start_idx[s_amino_seq_idx[amino_seq_idx]] + pos) * CODON_SIZE;

                s_solution[j] = c_codons[k];
                s_solution[j + 1] = c_codons[k + 1];
                s_solution[j + 2] = c_codons[k + 2];
            }
        }
        break;

    case HIGHEST_CAI_GEN:
        for (i = 0; i < partition_num; i++)
        {
            idx = blockDim.x * i + threadIdx.x;
            if (idx < c_amino_seq_len * c_cds_num)
            {
                amino_seq_idx = idx % c_amino_seq_len;

                pos = c_syn_codons_num[s_amino_seq_idx[amino_seq_idx]] - 1;

                j = idx * CODON_SIZE;
                k = (c_codons_start_idx[s_amino_seq_idx[amino_seq_idx]] + pos) * CODON_SIZE;

                s_solution[j] = c_codons[k];
                s_solution[j + 1] = c_codons[k + 1];
                s_solution[j + 2] = c_codons[k + 2];
            }
        }
        break;
    }

    return;
}


#endif

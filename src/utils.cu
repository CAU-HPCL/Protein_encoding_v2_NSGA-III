#include <stdio.h>

#include <curand_kernel.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

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

            if ((j % 2 == 1) && (tb.thread_rank() == 0)) // 홀수일 경우 대비한 코드
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

    partition_num = ((c_amino_seq_len - 1) % tb.size() == 0) ? ((c_amino_seq_len - 1) / tb.size()) : ((c_amino_seq_len - 1) / tb.size()) + 1;
    for (i = 0; i < c_cds_num; i++)
    {
        s_obj_buffer[tb.thread_rank()] = 0.f;

        for (j = 0; j < partition_num; j++)
        {
            amino_seq_idx = tb.size() * j + tb.thread_rank();
            aminoacid_idx = s_amino_seq_idx[amino_seq_idx];
            right_aminoacid_idx = s_amino_seq_idx[amino_seq_idx + 1];
            if (amino_seq_idx < (c_amino_seq_len - 1))
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

            if ((j % 2 == 1) && (tb.thread_rank() == 0)) // 홀수일 경우 대비한 코드
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
                s_obj_val[MIN_CBP_IDX] = s_obj_buffer[0];
                s_obj_idx[MIN_CBP_IDX * 2] = i;
            }
            else if (s_obj_buffer[0] <= s_obj_val[MIN_CBP_IDX])
            {
                s_obj_val[MIN_CBP_IDX] = s_obj_buffer[0];
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

            if ((j % 2 == 1) && (tb.thread_rank() == 0)) // 홀수일 경우 대비한 코드
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
                    s_obj_buffer[threadIdx.x] += 1;
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
                    break;
            }

            if ((k % 2 == 1) && (threadIdx.x == 0))
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

    return;
}

__device__ void calMaximumGC(const thread_block tb, const char *solution, const char *s_amino_seq_idx, float *s_obj_buffer, float *s_obj_val, char *s_obj_idx)
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

                if ((base_idx < c_cds_len) && (solution[c_cds_len * i + base_idx] != ptr_target_sol[c_cds_len * j + base_idx]))
                {
                    s_obj_buffer[threadIdx.x] += 1;
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
                    break;
            }

            if ((k % 2 == 1) && (threadIdx.x == 0))
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


    return;
}

__device__ void calMaximumSL()
{

    return;
}
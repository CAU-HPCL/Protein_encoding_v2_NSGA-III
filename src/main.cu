#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <python3.10/Python.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda.h>

#include "../include/common.cuh"
#include "../include/info.cuh"
#include "../include/utils.cuh"
#include "../include/mutation.cuh"
#include "../include/sorting.cuh"

#define _CRT_SECURE_NO_WARINGS

using namespace cooperative_groups;

float h_true_ideal_value[OBJECTIVE_NUM] = {1.f, 0.732581f, 1.f, 0.4f, 0.f, 0.f};
float h_true_nadir_value[OBJECTIVE_NUM] = {0.f, -0.985957f, 0.f, 0.f, 0.6f, 1.f};

__global__ void initializationKernel(curandStateXORWOW *random_generator, unsigned long long seed, const char *d_amino_seq_idx, char *d_population, float *d_obj_val, char *d_obj_idx, int *d_pql, int *d_sorted_array)
{
    auto g = this_grid();
    auto tb = this_thread_block();
    int i;
    int idx;
    int partition_num;
    int cycle_partition_num;

    curand_init(seed, g.thread_rank(), 0, &random_generator[g.thread_rank()]);

    extern __shared__ int smem[];
    __shared__ int *s_pql;
    __shared__ int *s_mutex;
    __shared__ float *s_obj_val;
    __shared__ float *s_obj_buffer;
    __shared__ char *s_amino_seq_idx;
    __shared__ char *s_solution;
    __shared__ char *s_obj_idx;

    s_pql = smem;
    s_mutex = (int *)&s_pql[3];
    s_obj_val = (float *)&s_mutex[1];
    s_obj_buffer = (float *)&s_obj_val[OBJECTIVE_NUM];
    s_amino_seq_idx = (char *)&s_obj_buffer[tb.size()];
    s_solution = (char *)&s_amino_seq_idx[c_amino_seq_len];
    s_obj_idx = (char *)&s_solution[c_solution_len];

    curandStateXORWOW local_generator = random_generator[g.thread_rank()];
    partition_num = (c_amino_seq_len % tb.size() == 0) ? (c_amino_seq_len / tb.size()) : (c_amino_seq_len / tb.size()) + 1;
    for (i = 0; i < partition_num; i++)
    {
        idx = tb.size() * i + tb.thread_rank();
        if (idx < c_amino_seq_len)
        {
            s_amino_seq_idx[idx] = d_amino_seq_idx[idx];
        }
    }
    tb.sync();

    cycle_partition_num = (c_N % g.num_blocks() == 0) ? (c_N / g.num_blocks()) : (c_N / g.num_blocks()) + 1;
    for (i = 0; i < cycle_partition_num; i++)
    {
        idx = g.num_blocks() * i + g.block_rank();
        if (idx < c_N)
        {
            if (idx == 0)
            {
                genPopulation(tb, &local_generator, s_amino_seq_idx, s_solution, HIGHEST_CAI_GEN);
            }
            else
            {
                genPopulation(tb, &local_generator, s_amino_seq_idx, s_solution, RANDOM_GEN);
            }
            tb.sync();

            calMinimumCAI(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMinimumCBP(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMinimumHSC(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMinimumHD(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMaximumGC(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMaximumSL(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx, s_pql, s_mutex);
            tb.sync();

            copySolution(tb, s_solution, s_obj_val, s_obj_idx, s_pql, &d_population[c_solution_len * idx], &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx], &d_pql[3 * idx]);
            tb.sync();

            d_sorted_array[idx] = idx;
        }
    }
    g.sync();

    if ((g.block_rank() < OBJECTIVE_NUM) && tb.thread_rank() < OBJECTIVE_NUM)
    {
        extreme_points[g.block_rank()][tb.thread_rank()] = s_obj_val[tb.thread_rank()];
    }

    if (g.thread_rank() == 0)
    {
        d_cur_cycle_num = 0;
    }

    random_generator[g.thread_rank()] = local_generator;

    return;
}

__global__ void mutationKernel(curandStateXORWOW *random_generator, const char *d_amino_seq_idx, char *d_population, float *d_obj_val, char *d_obj_idx, int *d_pql, const char *d_tmp_population, const float *d_tmp_obj_val, const char *d_tmp_obj_idx, const int *d_tmp_pql, const int *d_sorted_array)
{
    auto g = this_grid();
    auto tb = this_thread_block();
    int i;
    int idx;
    int partition_num;
    int cycle_partition_num;
    curandStateXORWOW local_generator = random_generator[g.thread_rank()];

    extern __shared__ int smem[];
    __shared__ int *s_pql;
    __shared__ int *s_mutex;
    __shared__ int *s_termination_check;
    __shared__ float *s_obj_val;
    __shared__ float *s_obj_buffer;
    __shared__ char *s_amino_seq_idx;
    __shared__ char *s_solution;
    __shared__ char *s_obj_idx;
    __shared__ char *s_mutation_type;

    s_pql = smem;
    s_mutex = (int *)&s_pql[3];
    s_termination_check = (int *)&s_mutex[1];
    s_obj_val = (float *)&s_termination_check[1];
    s_obj_buffer = (float *)&s_obj_val[OBJECTIVE_NUM];
    s_amino_seq_idx = (char *)&s_obj_buffer[tb.size()];
    s_solution = (char *)&s_amino_seq_idx[c_amino_seq_len];
    s_obj_idx = (char *)&s_solution[c_solution_len];
    s_mutation_type = (char *)&s_obj_idx[OBJECTIVE_NUM * 2];

    partition_num = (c_amino_seq_len % tb.size() == 0) ? (c_amino_seq_len / tb.size()) : (c_amino_seq_len / tb.size()) + 1;
    for (i = 0; i < partition_num; i++)
    {
        idx = tb.size() * i + tb.thread_rank();
        if (idx < c_amino_seq_len)
        {
            s_amino_seq_idx[idx] = d_amino_seq_idx[idx];
        }
    }
    tb.sync();

    cycle_partition_num = (c_N % g.num_blocks() == 0) ? (c_N / g.num_blocks()) : (c_N / g.num_blocks()) + 1;
    for (i = 0; i < cycle_partition_num; i++)
    {
        idx = g.num_blocks() * i + g.block_rank();
        if (idx < c_N)
        {
            copySolution(tb, &d_tmp_population[c_solution_len * d_sorted_array[idx]], &d_tmp_obj_val[OBJECTIVE_NUM * d_sorted_array[idx]], &d_tmp_obj_idx[OBJECTIVE_NUM * 2 * d_sorted_array[idx]], &d_tmp_pql[3 * d_sorted_array[idx]], s_solution, s_obj_val, s_obj_idx, s_pql);
            copySolution(tb, s_solution, s_obj_val, s_obj_idx, s_pql, &d_population[c_solution_len * idx], &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx], &d_pql[3 * idx]);

            if (tb.thread_rank() == 0)
            {
                do
                {
                    *s_mutation_type = (char)(curand_uniform(&local_generator) * (OBJECTIVE_NUM + 1));
                } while (*s_mutation_type == (OBJECTIVE_NUM + 1));
            }
            tb.sync();

            switch (*s_mutation_type)
            {
            case 0:
                mutationRandom(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_idx);
                break;
            case 1:
                mutationCAI(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_idx, SELECT_UPPER_RANDOM);
                break;
            case 2:
                mutationCBP(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_idx, SELECT_UPPER_RANDOM, d_cur_cycle_num % 2);
                break;
            case 3:
                mutationHSC(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_idx, SELECT_UPPER_RANDOM, d_cur_cycle_num % 2);
                break;
            case 4:
                mutationHD(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_idx);
                break;
            case 5:
                if (s_obj_idx[MAX_GC_IDX * 2 + 1] == GC_UP)
                {
                    mutationGC(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_val, s_obj_idx, SELECT_HIGH_GC);
                }
                else if (s_obj_idx[MAX_GC_IDX * 2 + 1] == GC_DOWN)
                {
                    mutationGC(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_val, s_obj_idx, SELECT_LOW_GC);
                }
                break;
            case 6:
                mutationSL(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx, s_pql, s_mutex, s_termination_check);
                break;
            }
            tb.sync();

            calMinimumCAI(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMinimumCBP(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMinimumHSC(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMinimumHD(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMaximumGC(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            tb.sync();
            calMaximumSL(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx, s_pql, s_mutex);
            tb.sync();

            copySolution(tb, s_solution, s_obj_val, s_obj_idx, s_pql, &d_population[c_solution_len * (c_N + idx)], &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], &d_pql[3 * (c_N + idx)]);
            tb.sync();
        }
    }

    if (g.thread_rank() == 0)
    {
        d_cur_cycle_num += 1;
    }

    random_generator[g.thread_rank()] = local_generator;

    return;
}

__global__ void globalInitializationKernel(curandStateXORWOW *random_generator, unsigned long long seed, const char *d_amino_seq_idx, char *d_population, float *d_obj_val, char *d_obj_idx, int *d_pql, int *d_sorted_array)
{
    auto g = this_grid();
    auto tb = this_thread_block();
    int i;
    int idx;
    int cycle_partition_num;

    curand_init(seed, g.thread_rank(), 0, &random_generator[g.thread_rank()]);

    extern __shared__ int smem[];
    __shared__ int *s_mutex;
    __shared__ float *s_obj_buffer;

    s_mutex = smem;
    s_obj_buffer = (float *)&s_mutex[1];

    curandStateXORWOW local_generator = random_generator[g.thread_rank()];

    cycle_partition_num = (c_N % g.num_blocks() == 0) ? (c_N / g.num_blocks()) : (c_N / g.num_blocks()) + 1;
    for (i = 0; i < cycle_partition_num; i++)
    {
        idx = g.num_blocks() * i + g.block_rank();
        if (idx < c_N)
        {
            if (idx == 0)
            {
                genPopulation(tb, &local_generator, d_amino_seq_idx, &d_population[c_solution_len * idx], HIGHEST_CAI_GEN);
            }
            else
            {
                genPopulation(tb, &local_generator, d_amino_seq_idx, &d_population[c_solution_len * idx], RANDOM_GEN);
            }
            tb.sync();

            calMinimumCAI(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            tb.sync();
            calMinimumCBP(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            tb.sync();
            calMinimumHSC(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            tb.sync();
            calMinimumHD(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            tb.sync();
            calMaximumGC(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            tb.sync();
            calMaximumSL(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx], &d_pql[3 * idx], s_mutex);
            tb.sync();

            d_sorted_array[idx] = idx;
        }
    }
    g.sync();

    if ((g.block_rank() < OBJECTIVE_NUM) && tb.thread_rank() < OBJECTIVE_NUM)
    {
        extreme_points[g.block_rank()][i] = d_obj_val[g.block_rank() * OBJECTIVE_NUM + tb.thread_rank()];
    }

    if (g.thread_rank() == 0)
    {
        d_cur_cycle_num = 0;
    }

    random_generator[g.thread_rank()] = local_generator;

    return;
}

__global__ void globalMutationKernel(curandStateXORWOW *random_generator, const char *d_amino_seq_idx, char *d_population, float *d_obj_val, char *d_obj_idx, int *d_pql, const char *d_tmp_population, const float *d_tmp_obj_val, const char *d_tmp_obj_idx, const int *d_tmp_pql, const int *d_sorted_array)
{
    auto g = this_grid();
    auto tb = this_thread_block();
    int i;
    int idx;
    int cycle_partition_num;
    curandStateXORWOW local_generator = random_generator[g.thread_rank()];

    extern __shared__ int smem[];
    __shared__ int *s_mutex;
    __shared__ int *s_termination_check;
    __shared__ float *s_obj_buffer;
    __shared__ char *s_mutation_type;

    s_mutex = smem;
    s_termination_check = (int *)&s_mutex[1];
    s_obj_buffer = (float *)&s_termination_check[1];
    s_mutation_type = (char *)&s_obj_buffer[tb.size()];

    cycle_partition_num = (c_N % g.num_blocks() == 0) ? (c_N / g.num_blocks()) : (c_N / g.num_blocks()) + 1;
    for (i = 0; i < cycle_partition_num; i++)
    {
        idx = g.num_blocks() * i + g.block_rank();
        if (idx < c_N)
        {
            copySolution(tb, &d_tmp_population[c_solution_len * d_sorted_array[idx]], &d_tmp_obj_val[OBJECTIVE_NUM * d_sorted_array[idx]], &d_tmp_obj_idx[OBJECTIVE_NUM * 2 * d_sorted_array[idx]], &d_tmp_pql[3 * d_sorted_array[idx]], &d_population[c_solution_len * idx], &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx], &d_pql[3 * idx]);
            copySolution(tb, &d_population[c_solution_len * idx], &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx], &d_pql[3 * idx], &d_population[c_solution_len * (c_N + idx)], &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], &d_pql[3 * (c_N + idx)]);

            if (tb.thread_rank() == 0)
            {
                do
                {
                    *s_mutation_type = (char)(curand_uniform(&local_generator) * (OBJECTIVE_NUM + 1));
                } while (*s_mutation_type == (OBJECTIVE_NUM + 1));
            }
            tb.sync();

            switch (*s_mutation_type)
            {
            case 0:
                mutationRandom(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
                break;
            case 1:
                mutationCAI(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], SELECT_UPPER_RANDOM);
                break;
            case 2:
                mutationCBP(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], SELECT_UPPER_RANDOM, d_cur_cycle_num % 2);
                break;
            case 3:
                mutationHSC(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], SELECT_UPPER_RANDOM, d_cur_cycle_num % 2);
                break;
            case 4:
                mutationHD(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
                break;
            case 5:
                if (d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx) + MAX_GC_IDX * 2 + 1] == GC_UP)
                {
                    mutationGC(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], SELECT_HIGH_GC);
                }
                else if (d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx) + MAX_GC_IDX * 2 + 1] == GC_DOWN)
                {
                    mutationGC(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], SELECT_LOW_GC);
                }
                break;
            case 6:
                mutationSL(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], &d_pql[3 * (c_N + idx)], s_mutex, s_termination_check);
                break;
            }
            tb.sync();

            calMinimumCAI(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            tb.sync();
            calMinimumCBP(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            tb.sync();
            calMinimumHSC(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            tb.sync();
            calMinimumHD(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            tb.sync();
            calMaximumGC(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            tb.sync();
            calMaximumSL(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], &d_pql[3 * (c_N + idx)], s_mutex);
            tb.sync();
        }
    }

    if (g.thread_rank() == 0)
    {
        d_cur_cycle_num += 1;
    }

    random_generator[g.thread_rank()] = local_generator;

    return;
}

__global__ void sortingKernel(curandStateXORWOW *random_generator, const float *d_obj_val, int *d_sorted_array, bool *F_set, bool *Sp_set, int *d_np, int *d_rank_count, float *d_buffer, int *index_num, const float *d_reference_points, int *d_included_solution_num, int *d_not_included_solution_num, int *d_solution_index_for_sorting, float *d_dist_of_solution)
{
    auto g = this_grid();
    auto tb = this_thread_block();

    extern __shared__ int smem[];
    __shared__ int *s_index_num;
    __shared__ float *s_buffer;
    __shared__ float *s_normalized_obj_val;
    __shared__ curandStateXORWOW *s_generator;

    s_index_num = smem;
    s_buffer = (float *)&s_index_num[tb.size()];
    s_normalized_obj_val = (float *)&s_buffer[tb.size()];
    s_generator = (curandStateXORWOW *)&s_normalized_obj_val[OBJECTIVE_NUM];

    if (tb.thread_rank() == 0)
    {
        *s_generator = random_generator[g.block_rank()];
    }

    nonDominatedSorting(g, d_obj_val, d_sorted_array, F_set, Sp_set, d_np, d_rank_count);
    g.sync();

    updateIdealValue(g, d_obj_val, d_buffer, d_sorted_array, d_rank_count, index_num);
    g.sync();

    // updateNadirValue_MNDF(g, d_obj_val, d_buffer, d_sorted_array, d_rank_count, index_num);

    // updateNadirValue_ME(g, d_obj_val, d_buffer, index_num, d_sorted_array, d_rank_count);

    updateNadirValue_HYP(g, tb, d_obj_val, d_buffer, d_sorted_array, d_rank_count, index_num);
    g.sync();

    if (!N_cut_check)
    {
        referenceBasedSorting(s_generator, g, tb, d_obj_val, d_sorted_array, d_rank_count, d_reference_points, d_included_solution_num, d_not_included_solution_num, d_solution_index_for_sorting, d_dist_of_solution, s_buffer, s_index_num, s_normalized_obj_val);
    }
    g.sync();

    if (tb.thread_rank() == 0)
    {
        random_generator[g.block_rank()] = *s_generator;
    }

    return;
}

/*
argv[1] : Input file name
argv[2] : Population size (N)
argv[3] : Generation count (G)
argv[4] : Number of CDS
argv[5] : Mutation probability (Pm)

For example
../Protein_FASTA/Q5VZP5.fasta.txt  10 10 2 0.15
*/
int main(const int argc, const char *argv[])
{
    srand((unsigned int)time(NULL));
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::chrono::duration<double> sec;

    cudaDeviceProp deviceProp;
    int dev = 0;
    int maxbytes = 99328;
    int totalMultiProcessor;
    CHECK_CUDA(cudaSetDevice(dev))
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev))
    CHECK_CUDA(cudaDeviceGetAttribute(&totalMultiProcessor, cudaDevAttrMultiProcessorCount, dev))
    CHECK_CUDA(cudaFuncSetAttribute(initializationKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes))
    CHECK_CUDA(cudaFuncSetAttribute(mutationKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes))

    int population_size = atoi(argv[2]);
    int gen_cycle_num = atoi(argv[3]);
    char cds_num = (char)atoi(argv[4]);
    float mutation_prob = atof(argv[5]);
    if ((population_size <= 0) || (gen_cycle_num < 0) || (cds_num <= 1) || (mutation_prob < 0.f) || (mutation_prob > 1.f))
    {
        printf("Line : %d Please cheking input parameters.. \n", __LINE__);
        return EXIT_FAILURE;
    }

    FILE *fp;
    char buffer[256];
    char *amino_seq;
    int amino_seq_len, cds_len, solution_len;
    int idx;

    fp = fopen(argv[1], "r");
    if (fp == NULL)
    {
        printf("Line : %d Opening Protein FASTA format file is failed.. \n", __LINE__);
        return EXIT_FAILURE;
    }
    fseek(fp, 0, SEEK_END);
    amino_seq_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    fgets(buffer, 256, fp);
    amino_seq_len -= ftell(fp);
    amino_seq = (char *)malloc(sizeof(char) * (amino_seq_len + 1));
    idx = 0;
    while (!feof(fp))
    {
        char tmp = fgetc(fp);
        if (tmp != '\n')
        {
            amino_seq[idx++] = tmp;
        }
    }
    amino_seq[idx - 1] = 'Z';
    amino_seq[idx] = (char)NULL;
    amino_seq_len = idx;
    cds_len = amino_seq_len * CODON_SIZE;
    solution_len = cds_len * cds_num;
    fclose(fp);

    unsigned long long seed = (unsigned long long)rand();
    char *h_amino_seq_idx;
    char *h_population;
    float *h_obj_val;
    char *h_obj_idx;
    int *h_rank_count;
    int *h_sorted_array;
    float *h_reference_points;

    curandStateXORWOW *d_random_generator;
    cudaEvent_t d_start, d_end;
    unsigned long long *d_seed;
    char *d_amino_seq_idx;
    char *d_population;
    char *d_tmp_population;
    float *d_obj_val;
    float *d_tmp_obj_val;
    char *d_obj_idx;
    char *d_tmp_obj_idx;
    int *d_pql;
    int *d_tmp_pql;
    int *d_np;
    bool *d_F_set, *d_Sp_set;
    int *d_rank_count;
    int *d_sorted_array;
    int *d_index_num;
    float *d_buffer;
    float *d_reference_points;
    int *d_included_solution_num;
    int *d_not_included_solution_num;
    int *d_solution_index_for_sorting;
    float *d_dist_of_solution;

    h_amino_seq_idx = (char *)malloc(sizeof(char) * amino_seq_len);
    for (int i = 0; i < amino_seq_len; i++)
    {
        h_amino_seq_idx[i] = findAminoIndex(amino_seq[i]);
    }

#if 0
    int partition_num = 1;
    while(true)
    {
        if (combination(OBJECTIVE_NUM + partition_num - 1, partition_num) < population_size)
        {
            partition_num += 1;
        }else
        {
            break;
        }
    }
#endif
    int ref_points_num = population_size;
    h_reference_points = (float *)malloc(sizeof(float) * OBJECTIVE_NUM * ref_points_num);
    float ref_points_setting_time;
    start_time = std::chrono::system_clock::now();
    getReferencePointsEnergy(h_reference_points, OBJECTIVE_NUM, ref_points_num);
    end_time = std::chrono::system_clock::now();
    sec = end_time - start_time;
    ref_points_setting_time = static_cast<float>(sec.count());
    // getReferencePointsDasDennis(h_reference_points, OBJECTIVE_NUM, partition_num);

    /* TODO : 커널당 최적 쓰레드 개수필요함 */
    int initialization_blocks_num;
    int initialization_numBlocksPerSm;
    int initialization_threads_per_block = 128;
    int mutation_blocks_num;
    int mutation_numBlocksPerSm;
    int mutation_threads_per_block = 256;
    int global_initialization_blocks_num;
    int global_initialization_numBlocksPerSm;
    int global_initialization_threads_per_block = 256;
    int global_mutation_blocks_num;
    int global_mutation_numBlocksPerSm;
    int global_mutation_threads_per_block = 512;
    int sorting_blocks_num;
    int sorting_numBlocksPerSm;
    int sorting_threads_per_block = 256;

    size_t using_global_memory_size;
    size_t using_constant_memory_size = sizeof(char) + sizeof(codons_start_idx) + sizeof(syn_codons_num) + sizeof(codons) + sizeof(int) * 6 + sizeof(float) * 2 + sizeof(codons_weight) + sizeof(cps);
    size_t initialzation_shared_memory_size = sizeof(int) * 4 + sizeof(float) * (OBJECTIVE_NUM + initialization_threads_per_block) + sizeof(char) * (amino_seq_len + solution_len + (OBJECTIVE_NUM * 2));
    size_t mutation_shared_memory_size = sizeof(int) * 5 + sizeof(float) * (OBJECTIVE_NUM + mutation_threads_per_block) + sizeof(char) * (amino_seq_len + solution_len + (OBJECTIVE_NUM * 2) + 1);
    size_t global_initialzation_shared_memory_size = sizeof(float) * global_initialization_threads_per_block + sizeof(int);
    size_t global_mutation_shared_memory_size = sizeof(int) * 2 + sizeof(float) * global_mutation_threads_per_block + sizeof(char);
    size_t sorting_shared_memory_size = sizeof(int) * sorting_threads_per_block + sizeof(float) * (sorting_threads_per_block + OBJECTIVE_NUM) + sizeof(curandStateXORWOW);

    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&initialization_numBlocksPerSm, initializationKernel, initialization_threads_per_block, initialzation_shared_memory_size))
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mutation_numBlocksPerSm, mutationKernel, mutation_threads_per_block, mutation_shared_memory_size))
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&global_initialization_numBlocksPerSm, globalInitializationKernel, global_initialization_threads_per_block, global_initialzation_shared_memory_size))
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&global_mutation_numBlocksPerSm, globalMutationKernel, global_mutation_threads_per_block, global_mutation_shared_memory_size))
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sorting_numBlocksPerSm, sortingKernel, sorting_threads_per_block, sorting_shared_memory_size))

    initialization_blocks_num = (population_size < deviceProp.multiProcessorCount * initialization_numBlocksPerSm) ? population_size : deviceProp.multiProcessorCount * initialization_numBlocksPerSm;
    mutation_blocks_num = (population_size < deviceProp.multiProcessorCount * mutation_numBlocksPerSm) ? population_size : deviceProp.multiProcessorCount * mutation_numBlocksPerSm;
    global_initialization_blocks_num = (population_size < deviceProp.multiProcessorCount * global_initialization_numBlocksPerSm) ? population_size : deviceProp.multiProcessorCount * global_initialization_numBlocksPerSm;
    global_mutation_blocks_num = (population_size < deviceProp.multiProcessorCount * global_mutation_numBlocksPerSm) ? population_size : deviceProp.multiProcessorCount * global_mutation_numBlocksPerSm;
    sorting_blocks_num = deviceProp.multiProcessorCount * sorting_numBlocksPerSm;

    bool shared_vs_global = true;
    if (mutation_shared_memory_size > maxbytes)
    {
        shared_vs_global = false;
    }

    h_population = (char *)malloc(sizeof(char) * solution_len * population_size * 2);
    h_obj_val = (float *)malloc(sizeof(float) * OBJECTIVE_NUM * population_size * 2);
    h_obj_idx = (char *)malloc(sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2);
    h_rank_count = (int *)malloc(sizeof(int) * population_size * 2);
    h_sorted_array = (int *)malloc(sizeof(int) * population_size * 2);

    int shared_generator_num = (initialization_blocks_num * initialization_threads_per_block > mutation_blocks_num * mutation_threads_per_block) ? (initialization_blocks_num * initialization_threads_per_block) : (mutation_blocks_num * mutation_threads_per_block);
    int global_generator_num = (global_initialization_blocks_num * global_initialization_threads_per_block > global_mutation_blocks_num * global_mutation_threads_per_block) ? (global_initialization_blocks_num * global_initialization_threads_per_block) : (global_mutation_blocks_num * global_mutation_threads_per_block);

    CHECK_CUDA(cudaEventCreate(&d_start))
    CHECK_CUDA(cudaEventCreate(&d_end))
    if (shared_vs_global)
    {
        CHECK_CUDA(cudaMalloc((void **)&d_random_generator, sizeof(curandStateXORWOW) * shared_generator_num))
    }
    else
    {
        CHECK_CUDA(cudaMalloc((void **)&d_random_generator, sizeof(curandStateXORWOW) * global_generator_num))
    }
    CHECK_CUDA(cudaMalloc((void **)&d_amino_seq_idx, sizeof(char) * amino_seq_len))
    CHECK_CUDA(cudaMalloc((void **)&d_population, sizeof(char) * solution_len * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_tmp_population, sizeof(char) * solution_len * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_tmp_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_pql, sizeof(int) * 3 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_tmp_pql, sizeof(int) * 3 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_sorted_array, sizeof(int) * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_rank_count, sizeof(int) * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_np, sizeof(int) * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_included_solution_num, sizeof(int) * ref_points_num))
    CHECK_CUDA(cudaMalloc((void **)&d_not_included_solution_num, sizeof(int) * ref_points_num))
    CHECK_CUDA(cudaMalloc((void **)&d_solution_index_for_sorting, sizeof(int) * (ref_points_num * population_size * 2)))
    CHECK_CUDA(cudaMalloc((void **)&d_index_num, sizeof(int) * (population_size * 2 + OBJECTIVE_NUM)))
    CHECK_CUDA(cudaMalloc((void **)&d_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_tmp_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_buffer, sizeof(float) * (population_size * 2 + OBJECTIVE_NUM)))
    CHECK_CUDA(cudaMalloc((void **)&d_reference_points, sizeof(float) * (ref_points_num * OBJECTIVE_NUM)))
    CHECK_CUDA(cudaMalloc((void **)&d_dist_of_solution, sizeof(float) * (population_size * 2)))
    CHECK_CUDA(cudaMalloc((void **)&d_F_set, sizeof(bool) * population_size * 2 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_Sp_set, sizeof(bool) * population_size * 2 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_seed, sizeof(unsigned long long)))

    CHECK_CUDA(cudaMemcpy(d_seed, &seed, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_amino_seq_idx, h_amino_seq_idx, sizeof(char) * amino_seq_len, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_reference_points, h_reference_points, sizeof(float) * OBJECTIVE_NUM * ref_points_num, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpyToSymbol(c_ref_GC_percent, &ref_GC_percent, sizeof(float)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_codons_start_idx, codons_start_idx, sizeof(codons_start_idx)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_syn_codons_num, syn_codons_num, sizeof(syn_codons_num)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_codons, codons, sizeof(codons)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_codons_weight, codons_weight, sizeof(codons_weight)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_cps, cps, sizeof(cps)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_N, &population_size, sizeof(int)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_amino_seq_len, &amino_seq_len, sizeof(int)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_solution_len, &solution_len, sizeof(int)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_cds_len, &cds_len, sizeof(int)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_cds_num, &cds_num, sizeof(char)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_mutation_prob, &mutation_prob, sizeof(float)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_gen_cycle_num, &gen_cycle_num, sizeof(int)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_ref_points_num, &ref_points_num, sizeof(int)))

    float initialization_time = 0.f;
    float generation_cycle_time = 0.f;
    float total_time = 0.f;
    float min_dist = 0.f;
    void *initialization_args[] = {&d_random_generator, &d_seed, &d_amino_seq_idx, &d_population, &d_obj_val, &d_obj_idx, &d_pql, &d_sorted_array};
    void *even_mutation_args[] = {&d_random_generator, &d_amino_seq_idx, &d_tmp_population, &d_tmp_obj_val, &d_tmp_obj_idx, &d_tmp_pql, &d_population, &d_obj_val, &d_obj_idx, &d_pql, &d_sorted_array};
    void *even_sorting_args[] = {&d_random_generator, &d_tmp_obj_val, &d_sorted_array, &d_F_set, &d_Sp_set, &d_np, &d_rank_count, &d_buffer, &d_index_num, &d_reference_points, &d_included_solution_num, &d_not_included_solution_num, &d_solution_index_for_sorting, &d_dist_of_solution};
    void *odd_mutation_args[] = {&d_random_generator, &d_amino_seq_idx, &d_population, &d_obj_val, &d_obj_idx, &d_pql, &d_tmp_population, &d_tmp_obj_val, &d_tmp_obj_idx, &d_tmp_pql, &d_sorted_array};
    void *odd_sorting_args[] = {&d_random_generator, &d_obj_val, &d_sorted_array, &d_F_set, &d_Sp_set, &d_np, &d_rank_count, &d_buffer, &d_index_num, &d_reference_points, &d_included_solution_num, &d_not_included_solution_num, &d_solution_index_for_sorting, &d_dist_of_solution};

    using_global_memory_size = (sizeof(char) * (amino_seq_len + solution_len * population_size * 2 + OBJECTIVE_NUM * 2 * population_size * 2 + solution_len * population_size * 2 + OBJECTIVE_NUM * 2 * population_size * 2)) + (sizeof(int) * (3 * population_size * 2 + 3 * population_size * 2 + population_size * 2 + population_size * 2 + population_size * 2 + ref_points_num + ref_points_num + ref_points_num * population_size * 2 + population_size * 2 + OBJECTIVE_NUM + 5)) + (sizeof(float) * (OBJECTIVE_NUM * population_size * 2 + OBJECTIVE_NUM * population_size * 2 + population_size * 2 + OBJECTIVE_NUM + ref_points_num * OBJECTIVE_NUM + population_size * 2 + 1 + OBJECTIVE_NUM + OBJECTIVE_NUM + OBJECTIVE_NUM * OBJECTIVE_NUM + OBJECTIVE_NUM + OBJECTIVE_NUM * (OBJECTIVE_NUM + 1))) + (sizeof(bool) * (population_size * 2 * population_size * 2 + population_size * 2 * population_size * 2 + 2)) + sizeof(unsigned long long) + sizeof(curandStateXORWOW) * (shared_vs_global ? shared_generator_num : global_generator_num);
    printf("Global memory usage : %lu bytes\n", using_global_memory_size);
    printf("Constant memory usage : %lu bytes\n", using_constant_memory_size);
    printf("Initialzation Kernel Shared memory usage : %lu bytes\n", initialzation_shared_memory_size);
    printf("Mutation Kernel Shared memory usage : %lu bytes\n", mutation_shared_memory_size);
    printf("Global Initialzation Kernel Shared memory usage : %lu bytes\n", global_initialzation_shared_memory_size);
    printf("Global Mutation Kernel Shared memory usage : %lu bytes\n", global_mutation_shared_memory_size);
    printf("Sorting Kernel Shared memory usage : %lu bytes\n", sorting_shared_memory_size);
    if (shared_vs_global)
    {
        CHECK_CUDA(cudaEventRecord(d_start))
        CHECK_CUDA(cudaLaunchCooperativeKernel((void **)initializationKernel, initialization_blocks_num, initialization_threads_per_block, initialization_args, initialzation_shared_memory_size))
        CHECK_CUDA(cudaEventRecord(d_end))
        CHECK_CUDA(cudaEventSynchronize(d_end))
        CHECK_CUDA(cudaEventElapsedTime(&initialization_time, d_start, d_end))

        CHECK_CUDA(cudaEventRecord(d_start))
        for (int i = 0; i < gen_cycle_num; i++)
        {
            CHECK_CUDA(cudaMemset(d_F_set, false, sizeof(bool) * 2 * population_size * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_Sp_set, false, sizeof(bool) * 2 * population_size * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_rank_count, 0, sizeof(int) * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_np, 0, sizeof(int) * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_included_solution_num, 0, sizeof(int) * ref_points_num))
            CHECK_CUDA(cudaMemset(d_not_included_solution_num, 0, sizeof(int) * ref_points_num))
            CHECK_CUDA(cudaMemset(d_solution_index_for_sorting, EMPTY, sizeof(int) * 2 * ref_points_num * population_size))
            CHECK_CUDA(cudaMemset(d_dist_of_solution, EMPTY, sizeof(float) * 2 * population_size))
            if (i % 2 == 0)
            {
                CHECK_CUDA(cudaLaunchCooperativeKernel((void **)mutationKernel, mutation_blocks_num, mutation_threads_per_block, even_mutation_args, mutation_shared_memory_size))
                CHECK_CUDA(cudaLaunchCooperativeKernel((void **)sortingKernel, sorting_blocks_num, sorting_threads_per_block, even_sorting_args, sorting_shared_memory_size))
            }
            else
            {
                CHECK_CUDA(cudaLaunchCooperativeKernel((void **)mutationKernel, mutation_blocks_num, mutation_threads_per_block, odd_mutation_args, mutation_shared_memory_size))
                CHECK_CUDA(cudaLaunchCooperativeKernel((void **)sortingKernel, sorting_blocks_num, sorting_threads_per_block, odd_sorting_args, sorting_shared_memory_size))
            }
        }
        CHECK_CUDA(cudaEventRecord(d_end))
        CHECK_CUDA(cudaEventSynchronize(d_end))
        CHECK_CUDA(cudaEventElapsedTime(&generation_cycle_time, d_start, d_end))
    }
    else
    {
        CHECK_CUDA(cudaEventRecord(d_start))
        CHECK_CUDA(cudaLaunchCooperativeKernel((void **)globalInitializationKernel, global_initialization_blocks_num, global_initialization_threads_per_block, initialization_args, global_initialzation_shared_memory_size))
        CHECK_CUDA(cudaEventRecord(d_end))
        CHECK_CUDA(cudaEventSynchronize(d_end))
        CHECK_CUDA(cudaEventElapsedTime(&initialization_time, d_start, d_end))

        CHECK_CUDA(cudaEventRecord(d_start))
        for (int i = 0; i < gen_cycle_num; i++)
        {
            CHECK_CUDA(cudaMemset(d_F_set, false, sizeof(bool) * 2 * population_size * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_Sp_set, false, sizeof(bool) * 2 * population_size * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_rank_count, 0, sizeof(int) * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_np, 0, sizeof(int) * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_included_solution_num, 0, sizeof(int) * ref_points_num))
            CHECK_CUDA(cudaMemset(d_not_included_solution_num, 0, sizeof(int) * ref_points_num))
            CHECK_CUDA(cudaMemset(d_solution_index_for_sorting, EMPTY, sizeof(int) * 2 * ref_points_num * population_size))
            CHECK_CUDA(cudaMemset(d_dist_of_solution, EMPTY, sizeof(float) * 2 * population_size))

            if (i % 2 == 0)
            {
                CHECK_CUDA(cudaLaunchCooperativeKernel((void **)globalMutationKernel, global_mutation_blocks_num, global_mutation_threads_per_block, even_mutation_args, global_mutation_shared_memory_size))
                CHECK_CUDA(cudaLaunchCooperativeKernel((void **)sortingKernel, sorting_blocks_num, sorting_threads_per_block, even_sorting_args, sorting_shared_memory_size))
            }
            else
            {
                CHECK_CUDA(cudaLaunchCooperativeKernel((void **)globalMutationKernel, global_mutation_blocks_num, global_mutation_threads_per_block, odd_mutation_args, global_mutation_shared_memory_size))
                CHECK_CUDA(cudaLaunchCooperativeKernel((void **)sortingKernel, sorting_blocks_num, sorting_threads_per_block, odd_sorting_args, sorting_shared_memory_size))
            }
        }
        CHECK_CUDA(cudaEventRecord(d_end))
        CHECK_CUDA(cudaEventSynchronize(d_end))
        CHECK_CUDA(cudaEventElapsedTime(&generation_cycle_time, d_start, d_end))
    }
    initialization_time /= 1000.f;
    generation_cycle_time /= 1000.f;
    total_time = ref_points_setting_time + initialization_time + generation_cycle_time;

    if (gen_cycle_num % 2 == 0)
    {
        CHECK_CUDA(cudaMemcpy(h_population, d_population, sizeof(char) * solution_len * population_size * 2, cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(h_obj_val, d_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2, cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(h_obj_idx, d_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2, cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(h_rank_count, d_rank_count, sizeof(int) * population_size * 2, cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(h_sorted_array, d_sorted_array, sizeof(int) * population_size * 2, cudaMemcpyDeviceToHost))
    }
    else
    {
        CHECK_CUDA(cudaMemcpy(h_population, d_tmp_population, sizeof(char) * solution_len * population_size * 2, cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(h_obj_val, d_tmp_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2, cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(h_obj_idx, d_tmp_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2, cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(h_rank_count, d_rank_count, sizeof(int) * population_size * 2, cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(h_sorted_array, d_sorted_array, sizeof(int) * population_size * 2, cudaMemcpyDeviceToHost))
    }

    for (int i = 0; i < population_size * 2; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            h_obj_val[i * OBJECTIVE_NUM + j] = -h_obj_val[i * OBJECTIVE_NUM + j];
        }
    }

    fp = fopen("Result.txt", "w");
    for (int i = 0; i < population_size * 2; i++)
    {
        fprintf(fp, "%d Solutions\n", i + 1);
        for (int j = 0; j < cds_num; j++)
        {
            fprintf(fp, "%d CDS : ", j + 1);
            for (int k = 0; k < cds_len; k++)
            {
                fprintf(fp, "%c", h_population[solution_len * i + cds_len * j + k]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n %d Solution\n", i + 1);
        fprintf(fp, "%d\tmCAI : %f\n", h_obj_idx[i * (OBJECTIVE_NUM * 2) + MIN_CAI_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MIN_CAI_IDX]);
        fprintf(fp, "%d\tmCBP : %f\n", h_obj_idx[i * (OBJECTIVE_NUM * 2) + MIN_CBP_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MIN_CBP_IDX]);
        fprintf(fp, "%d\tmHSC : %f\n", h_obj_idx[i * (OBJECTIVE_NUM * 2) + MIN_HSC_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MIN_HSC_IDX]);
        fprintf(fp, "%d %d\tmHD : %f\n", h_obj_idx[i * (OBJECTIVE_NUM * 2) + MIN_HD_IDX * 2], h_obj_idx[i * (OBJECTIVE_NUM * 2) + MIN_HD_IDX * 2 + 1], h_obj_val[i * OBJECTIVE_NUM + MIN_HD_IDX]);
        fprintf(fp, "%d\tMGC : %f\n", h_obj_idx[i * (OBJECTIVE_NUM * 2) + MAX_GC_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MAX_GC_IDX]);
        fprintf(fp, "%d\tMSL : %f\n", h_obj_idx[i * (OBJECTIVE_NUM * 2) + MAX_SL_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MAX_SL_IDX]);
    }
    fclose(fp);

    for (int i = 0; i < population_size * 2; i++)
    {
        for (int j = 0; j < OBJECTIVE_NUM; j++)
        {
            h_obj_val[i * OBJECTIVE_NUM + j] = (h_obj_val[i * OBJECTIVE_NUM + j] - h_true_ideal_value[j]) / (h_true_nadir_value[j] - h_true_ideal_value[j]);
            if(h_obj_val[i * OBJECTIVE_NUM + j] == 1)
            {
                h_obj_val[i * OBJECTIVE_NUM + j] -= 0.000001f;
            }
        }
    }

    min_dist = MinEuclid(h_obj_val, population_size * 2);

    fp = fopen("Normalized_value_quality_computation.txt", "w");
    for (int i = 0; i < population_size * 2; i++)
    {
        fprintf(fp, "%f %f %f %f %f %f\n", h_obj_val[i * OBJECTIVE_NUM + MIN_CAI_IDX], h_obj_val[i * OBJECTIVE_NUM + MIN_CBP_IDX], h_obj_val[i * OBJECTIVE_NUM + MIN_HSC_IDX], h_obj_val[i * OBJECTIVE_NUM + MIN_HD_IDX], h_obj_val[i * OBJECTIVE_NUM + MAX_GC_IDX], h_obj_val[i * OBJECTIVE_NUM + MAX_SL_IDX]);
    }
    fclose(fp);

    char command[100] = "./hv -r \"1 1 1 1 1 1\" Normalized_value_quality_computation.txt";
    FILE *pipe = popen(command, "r");
    if (!pipe)
    {
        printf("Failed to Execute Hypervolume execution program\n");
        return EXIT_FAILURE;
    }

    while (fgets(buffer, sizeof(buffer), pipe))
    {
        printf("\n\nHypervolume : %s", buffer);
    }
    pclose(pipe);
    printf("Minimum Distance to Ideal Point : %f\n", min_dist);
    printf("Ref points setting time : %f seconds\n", ref_points_setting_time);
    printf("Initialization time : %f seconds\n", initialization_time);
    printf("Generation cycles time : %f seconds\n", generation_cycle_time);
    printf("Total time : %f seconds\n", total_time);

    free(amino_seq);
    free(h_amino_seq_idx);
    free(h_population);
    free(h_obj_val);
    free(h_obj_idx);
    free(h_reference_points);
    free(h_rank_count);
    free(h_sorted_array);

    CHECK_CUDA(cudaEventDestroy(d_start))
    CHECK_CUDA(cudaEventDestroy(d_end))
    CHECK_CUDA(cudaFree(d_random_generator))
    CHECK_CUDA(cudaFree(d_seed))
    CHECK_CUDA(cudaFree(d_amino_seq_idx))
    CHECK_CUDA(cudaFree(d_population))
    CHECK_CUDA(cudaFree(d_tmp_population))
    CHECK_CUDA(cudaFree(d_obj_val))
    CHECK_CUDA(cudaFree(d_tmp_obj_val))
    CHECK_CUDA(cudaFree(d_obj_idx))
    CHECK_CUDA(cudaFree(d_tmp_obj_idx))
    CHECK_CUDA(cudaFree(d_pql))
    CHECK_CUDA(cudaFree(d_tmp_pql))
    CHECK_CUDA(cudaFree(d_sorted_array))
    CHECK_CUDA(cudaFree(d_np))
    CHECK_CUDA(cudaFree(d_F_set))
    CHECK_CUDA(cudaFree(d_Sp_set))
    CHECK_CUDA(cudaFree(d_rank_count))
    CHECK_CUDA(cudaFree(d_buffer))
    CHECK_CUDA(cudaFree(d_index_num))
    CHECK_CUDA(cudaFree(d_reference_points))
    CHECK_CUDA(cudaFree(d_included_solution_num))
    CHECK_CUDA(cudaFree(d_not_included_solution_num))
    CHECK_CUDA(cudaFree(d_solution_index_for_sorting))
    CHECK_CUDA(cudaFree(d_dist_of_solution))

    return EXIT_SUCCESS;
}

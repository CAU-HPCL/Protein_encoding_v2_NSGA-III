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

__device__ int d_cur_cycle_num;

/* Using shared memory */
__global__ void initializationKernel(curandStateXORWOW *random_generator, unsigned long long seed, const char *d_amino_seq_idx, char *d_population, float *d_obj_val, char *d_obj_idx, int *d_pql, int *d_sorted_array)
{
    auto g = this_grid();
    auto tb = this_thread_block();
    int i;
    int idx;
    int partition_num;
    int cycle_partition_num;

    curand_init(seed, g.thread_rank(), 0, &random_generator[g.thread_rank()]); // Initialization of random generator

    /* Shared memory allocation */
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
    /* Solutions initialization */
    for (i = 0; i < cycle_partition_num; i++)
    {
        idx = g.num_blocks() * i + g.block_rank();
        if (idx < c_N)
        {
            if (idx == (c_N - 1))
            {
                genPopulation(tb, &local_generator, s_amino_seq_idx, s_solution, HIGHEST_CAI_GEN);
            }
            else
            {
                genPopulation(tb, &local_generator, s_amino_seq_idx, s_solution, RANDOM_GEN);
            }

            /* Calculating objective function (+ 논문에 따라 추가적인 정규화 작업이 필요할 수 있음)*/
            calMinimumCAI(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMinimumCBP(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMinimumHSC(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMinimumHD(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMaximumGC(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMaximumSL(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx, s_pql, s_mutex);

            copySolution(tb, s_solution, s_obj_val, s_obj_idx, s_pql, &d_population[c_solution_len * idx], &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx], &d_pql[3 * idx]);

            d_sorted_array[idx] = idx;
        }
    }

    random_generator[g.thread_rank()] = local_generator;

    if (g.thread_rank() == 0)
    {
        d_cur_cycle_num = 0;
    }

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

    /* Shared memory allocation */
    extern __shared__ int smem[];
    __shared__ int *s_pql;
    __shared__ int *s_mutex;
    __shared__ int *s_proceed_check;
    __shared__ int *s_termination_check;
    __shared__ float *s_obj_val;
    __shared__ float *s_obj_buffer;
    __shared__ char *s_amino_seq_idx;
    __shared__ char *s_solution;
    __shared__ char *s_obj_idx;
    __shared__ char *s_mutation_type;

    s_pql = smem;
    s_mutex = (int *)&s_pql[3];
    s_proceed_check = (int *)&s_mutex[1];
    s_termination_check = (int *)&s_proceed_check[1];
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

            /* Mutation */
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
                mutationGC(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_idx, SELECT_LOW_GC); // 여기는 low high 추가적인 조치가 필요함
                break;
            case 6:
                mutationSL(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx, s_pql, s_mutex, s_proceed_check, s_termination_check);
                break;
            }
            tb.sync();

            /* Calculating objective function (+ 논문에 따라 추가적인 정규화 작업이 필요할 수 있음) */
            calMinimumCAI(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMinimumCBP(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMinimumHSC(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMinimumHD(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMaximumGC(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx);
            calMaximumSL(tb, s_solution, s_amino_seq_idx, s_obj_buffer, s_obj_val, s_obj_idx, s_pql, s_mutex);

            copySolution(tb, s_solution, s_obj_val, s_obj_idx, s_pql, &d_population[c_solution_len * (c_N + idx)], &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], &d_pql[3 * (c_N + idx)]);
        }
    }

    random_generator[g.thread_rank()] = local_generator;

    if (g.thread_rank() == 0)
    {
        d_cur_cycle_num += 1;
    }

    return;
}

/* Using global memory */
__global__ void globalInitializationKernel(curandStateXORWOW *random_generator, unsigned long long seed, const char *d_amino_seq_idx, char *d_population, float *d_obj_val, char *d_obj_idx, int *d_pql, int *d_sorted_array)
{
    auto g = this_grid();
    auto tb = this_thread_block();
    int i;
    int idx;
    int cycle_partition_num;

    curand_init(seed, g.thread_rank(), 0, &random_generator[g.thread_rank()]); // Initialization of random generator

    /* Shared memory allocation */
    extern __shared__ int smem[];
    __shared__ int *s_mutex;
    __shared__ float *s_obj_buffer;

    s_mutex = smem;
    s_obj_buffer = (float *)&s_mutex[1];

    curandStateXORWOW local_generator = random_generator[g.thread_rank()];

    cycle_partition_num = (c_N % g.num_blocks() == 0) ? (c_N / g.num_blocks()) : (c_N / g.num_blocks()) + 1;
    /* Solutions initialization */
    for (i = 0; i < cycle_partition_num; i++)
    {
        idx = g.num_blocks() * i + g.block_rank();
        if (idx < c_N)
        {
            if (idx == (c_N - 1))
            {
                genPopulation(tb, &local_generator, d_amino_seq_idx, &d_population[c_solution_len * idx], HIGHEST_CAI_GEN);
            }
            else
            {
                genPopulation(tb, &local_generator, d_amino_seq_idx, &d_population[c_solution_len * idx], RANDOM_GEN);
            }

            /* Calculating objective function (+ 논문에 따라 추가적인 정규화 작업이 필요할 수 있음)*/
            calMinimumCAI(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            calMinimumCBP(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            calMinimumHSC(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            calMinimumHD(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            calMaximumGC(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx]);
            calMaximumSL(tb, &d_population[c_solution_len * idx], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * idx], &d_obj_idx[OBJECTIVE_NUM * 2 * idx], &d_pql[3 * idx], s_mutex);

            d_sorted_array[idx] = idx;
        }
    }

    random_generator[g.thread_rank()] = local_generator;

    if (g.thread_rank() == 0)
    {
        d_cur_cycle_num = 0;
    }

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

    /* Shared memory allocation */
    extern __shared__ int smem[];
    __shared__ int *s_mutex;
    __shared__ int *s_proceed_check;
    __shared__ int *s_termination_check;
    __shared__ float *s_obj_buffer;
    __shared__ char *s_mutation_type;

    s_mutex = smem;
    s_proceed_check = (int *)&s_mutex[1];
    s_termination_check = (int *)&s_proceed_check[1];
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

            /* Mutation */
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
                mutationGC(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], SELECT_LOW_GC); // 여기는 low high 추가적인 조치가 필요함
                break;
            case 6:
                mutationSL(tb, &local_generator, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], &d_pql[3 * (c_N + idx)], s_mutex, s_proceed_check, s_termination_check);
                break;
            }
            tb.sync();

            /* Calculating objective function (+ 논문에 따라 추가적인 정규화 작업이 필요할 수 있음) */
            calMinimumCAI(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            calMinimumCBP(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            calMinimumHSC(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            calMinimumHD(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            calMaximumGC(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)]);
            calMaximumSL(tb, &d_population[c_solution_len * (c_N + idx)], d_amino_seq_idx, s_obj_buffer, &d_obj_val[OBJECTIVE_NUM * (c_N + idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + idx)], &d_pql[3 * (c_N + idx)], s_mutex);
        }
    }

    random_generator[g.thread_rank()] = local_generator;

    if (g.thread_rank() == 0)
    {
        d_cur_cycle_num += 1;
    }

    return;
}

__global__ void sortingKernel(float *d_obj_val, int *d_sorted_array, bool *F_set, bool *Sp_set, int *d_np, int *d_rank_count)
{
    auto g = this_grid();

    nonDominatedSorting(g, d_obj_val, d_sorted_array, F_set, Sp_set, d_np, d_rank_count);

    return;
}

/*
argv[1] : Input file name
argv[2] : Population size (N)
argv[3] : Generation count (G)
argv[4] : Number of CDS
argv[5] : Mutation probability (Pm)

For example
../Protein_FASTA/Q5VZP5.fasta.txt  10 10 2 0.5 32
*/
// 공유 메모리 커널당 최대치 설정하도록 바꾸는거 참조 필요
int main(const int argc, const char *argv[])
{
    srand((unsigned int)time(NULL));

    /* Getting information of Deivce */
    cudaDeviceProp deviceProp;
    int dev = 0;
    int maxSharedMemPerBlock;
    int maxSharedMemPerProcessor;
    int totalConstantMem;
    int maxRegisterPerProcessor;
    int maxRegisterPerBlock;
    int totalMultiProcessor;

    CHECK_CUDA(cudaSetDevice(dev))
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev))
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, dev))
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemPerProcessor, cudaDevAttrMaxRegistersPerMultiprocessor, dev))
    CHECK_CUDA(cudaDeviceGetAttribute(&totalConstantMem, cudaDevAttrTotalConstantMemory, dev))
    CHECK_CUDA(cudaDeviceGetAttribute(&maxRegisterPerProcessor, cudaDevAttrMaxRegistersPerMultiprocessor, dev))
    CHECK_CUDA(cudaDeviceGetAttribute(&maxRegisterPerBlock, cudaDevAttrMaxRegistersPerBlock, dev))
    CHECK_CUDA(cudaDeviceGetAttribute(&totalMultiProcessor, cudaDevAttrMultiProcessorCount, dev))

    printf("Device #%d:\n", dev);
    printf("Name: %s\n", deviceProp.name);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Clock rate: %d MHz\n", deviceProp.clockRate / 1000);
    printf("Global memory size: %lu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
    printf("Max thread dimensions: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Total constant memory: %d bytes\n", totalConstantMem);
    printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum shared memory per SM: %d bytes\n", maxSharedMemPerProcessor);
    printf("Maximum shared memory per block: %d bytes\n", maxSharedMemPerBlock);
    printf("Maximum number of registers per SM: %d\n", maxRegisterPerProcessor);
    printf("Maximum number of registers per block: %d\n", maxRegisterPerBlock);
    printf("Total number of SM in device: %d\n", totalMultiProcessor);
    printf("\n");

    /* Checking input parameters */
    int population_size = atoi(argv[2]);
    int gen_cycle_num = atoi(argv[3]);
    char cds_num = (char)atoi(argv[4]);
    float mutation_prob = atof(argv[5]);
    if ((population_size <= 0) || (gen_cycle_num < 0) || (cds_num <= 1) || (mutation_prob < 0.f) || (mutation_prob > 1.f))
    {
        printf("Line : %d Please cheking input parameters.. \n", __LINE__);
        return EXIT_FAILURE;
    }

    /* Preprocessing */
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
    amino_seq = (char *)malloc(sizeof(char) * (amino_seq_len + 1)); // +1 indicates last is stop codons.

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
    char *h_obj_idx;   // 나중에 제거
    int *h_rank_count; // 나중에 제거
    float *h_reference_points;

    curandStateXORWOW *d_random_generator;
    cudaEvent_t d_start, d_end;
    unsigned long long *d_seed;
    char *d_amino_seq_idx;
    char *d_population;
    float *d_obj_val;
    char *d_obj_idx;
    int *d_pql;

    // sorting 을 위해서 할당한 것들
    int *d_np;
    bool *d_F_set, *d_Sp_set;
    int *d_rank_count;
    int *d_sorted_array;
    char *d_tmp_population;
    float *d_tmp_obj_val;
    char *d_tmp_obj_idx;
    int *d_tmp_pql;

    h_amino_seq_idx = (char *)malloc(sizeof(char) * amino_seq_len);
    for (int i = 0; i < amino_seq_len; i++)
    {
        h_amino_seq_idx[i] = findAminoIndex(amino_seq[i]);
    }

    /* Setting Reference points */
    h_reference_points = (float *)malloc(sizeof(float) * OBJECTIVE_NUM * population_size);
    // getReferencePoints(h_reference_points, OBJECTIVE_NUM, population_size);      일단은 체크를 위해서 주석처리

    /* 커널의 블럭 당 쓰레드 개수는 나중에 추가적으로 다시 확인 필요한 부분 */
    int initialization_blocks_num;
    int initialization_numBlocksPerSm;
    int initialization_threads_per_block = 128;
    int mutation_blocks_num;
    int mutation_numBlocksPerSm;
    int mutation_threads_per_block = 512;
    int global_initialization_blocks_num;
    int global_initialization_numBlocksPerSm;
    int global_initialization_threads_per_block = 256;
    int global_mutation_blocks_num;
    int global_mutation_numBlocksPerSm;
    int global_mutation_threads_per_block = 512;
    int sorting_blocks_num;
    int sorting_numBlocksPerSm;
    int sorting_threads_per_block = 256;

    // size_t using_global_memory_size = sizeof(curandStateXORWOW) * (blocks_num * threads_per_block) + sizeof(unsigned long long) + sizeof(char) * (amino_seq_len + solution_len * population_size * 2 + OBJECTIVE_NUM * 2 * population_size * 2) + sizeof(float) * (OBJECTIVE_NUM * population_size * 2);    // 여기 계산 나중에 한번에 필요
    size_t using_constant_memory_size = sizeof(c_codons_start_idx) + sizeof(c_syn_codons_num) + sizeof(c_codons) + sizeof(c_codons_weight) + sizeof(c_cps) + sizeof(int) * 4 + sizeof(char) + sizeof(float);
    size_t initialzation_shared_memory_size = sizeof(float) * (OBJECTIVE_NUM + initialization_threads_per_block) + sizeof(char) * (amino_seq_len + solution_len + (OBJECTIVE_NUM * 2)) + sizeof(int) * 4;
    size_t mutation_shared_memory_size = sizeof(float) * (OBJECTIVE_NUM + mutation_threads_per_block) + sizeof(char) * (amino_seq_len + solution_len + (OBJECTIVE_NUM * 2) + 1) + sizeof(int) * 6;
    size_t global_initialzation_shared_memory_size = sizeof(float) * initialization_threads_per_block + sizeof(int);
    size_t global_mutation_shared_memory_size = sizeof(float) * mutation_threads_per_block + sizeof(char) + sizeof(int) * 3;
    size_t sorting_shared_memory_size = 0;

    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&initialization_numBlocksPerSm, initializationKernel, initialization_threads_per_block, initialzation_shared_memory_size))
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mutation_numBlocksPerSm, mutationKernel, mutation_threads_per_block, mutation_shared_memory_size))
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&global_initialization_numBlocksPerSm, globalInitializationKernel, global_initialization_threads_per_block, global_initialzation_shared_memory_size))
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&global_mutation_numBlocksPerSm, globalMutationKernel, global_mutation_threads_per_block, global_mutation_shared_memory_size))
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sorting_numBlocksPerSm, sortingKernel, sorting_threads_per_block, sorting_shared_memory_size))

    initialization_blocks_num = (population_size < deviceProp.multiProcessorCount * initialization_numBlocksPerSm) ? population_size : deviceProp.multiProcessorCount * initialization_numBlocksPerSm;
    mutation_blocks_num = (population_size < deviceProp.multiProcessorCount * mutation_numBlocksPerSm) ? population_size : deviceProp.multiProcessorCount * mutation_numBlocksPerSm;
    global_initialization_blocks_num = (population_size < deviceProp.multiProcessorCount * global_initialization_numBlocksPerSm) ? population_size : deviceProp.multiProcessorCount * global_initialization_numBlocksPerSm;
    global_mutation_blocks_num = (population_size < deviceProp.multiProcessorCount * global_mutation_numBlocksPerSm) ? population_size : deviceProp.multiProcessorCount * global_mutation_numBlocksPerSm;
    sorting_blocks_num = deviceProp.multiProcessorCount * sorting_numBlocksPerSm; // 여기도 나중에 다시 체크해야 할 부분

    bool shared_vs_global = true;
    if (mutation_shared_memory_size > maxSharedMemPerBlock) // true 면 shared memory 사용하는 부분으로 실행하게 됨
    {
        shared_vs_global = false;
    }

    // printf("다 짜고 다시 계산 필요 Global memory usage : %lu bytes\n", using_global_memory_size);
    printf("Constant memory usage : %lu bytes\n", using_constant_memory_size);
    printf("Initialzation Kernel Shared memory usage : %lu bytes\n", initialzation_shared_memory_size);
    printf("Mutation Kernel Shared memory usage : %lu bytes\n", mutation_shared_memory_size);
    printf("Global Initialzation Kernel Shared memory usage : %lu bytes\n", global_initialzation_shared_memory_size);
    printf("Global Mutation Kernel Shared memory usage : %lu bytes\n", global_mutation_shared_memory_size);
    printf("Sorting Kernel Shared memory usage : %lu bytes\n", sorting_shared_memory_size);

    /* Host Memory allocation */
    h_population = (char *)malloc(sizeof(char) * solution_len * population_size * 2);
    h_obj_val = (float *)malloc(sizeof(float) * OBJECTIVE_NUM * population_size * 2);
    h_obj_idx = (char *)malloc(sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2); // 나중에 제거
    h_rank_count = (int *)malloc(sizeof(int) * population_size * 2);                    // 나중에 제거

    /* Device Memory allocation */
    CHECK_CUDA(cudaEventCreate(&d_start))
    CHECK_CUDA(cudaEventCreate(&d_end))
    if (shared_vs_global)
    {
        CHECK_CUDA(cudaMalloc((void **)&d_random_generator, sizeof(curandStateXORWOW) * ((initialization_blocks_num > mutation_blocks_num) ? initialization_blocks_num : mutation_blocks_num) * ((initialization_threads_per_block > mutation_threads_per_block) ? initialization_threads_per_block : mutation_threads_per_block)))
    }
    else
    {
        CHECK_CUDA(cudaMalloc((void **)&d_random_generator, sizeof(curandStateXORWOW) * ((global_initialization_blocks_num > global_mutation_blocks_num) ? global_initialization_blocks_num : global_mutation_blocks_num) * ((global_initialization_threads_per_block > global_mutation_threads_per_block) ? global_initialization_threads_per_block : global_mutation_threads_per_block)))
    }
    CHECK_CUDA(cudaMalloc((void **)&d_seed, sizeof(unsigned long long)))
    CHECK_CUDA(cudaMalloc((void **)&d_amino_seq_idx, sizeof(char) * amino_seq_len))
    CHECK_CUDA(cudaMalloc((void **)&d_population, sizeof(char) * solution_len * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_pql, sizeof(int) * 3 * population_size * 2))

    CHECK_CUDA(cudaMalloc((void **)&d_tmp_population, sizeof(char) * solution_len * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_tmp_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_tmp_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_tmp_pql, sizeof(int) * 3 * population_size * 2))

    CHECK_CUDA(cudaMalloc((void **)&d_sorted_array, sizeof(int) * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_rank_count, sizeof(int) * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_np, sizeof(int) * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_F_set, sizeof(bool) * population_size * 2 * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_Sp_set, sizeof(bool) * population_size * 2 * population_size * 2))
    /* corwding distance sorting 안 할 거면 제거해야 하는 부분 */
    Sol *d_sol_struct;
    CHECK_CUDA(cudaMalloc((void **)&d_sol_struct, sizeof(Sol) * population_size * 2))


    /* Memory copy Host to Device */
    CHECK_CUDA(cudaMemcpy(d_seed, &seed, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_amino_seq_idx, h_amino_seq_idx, sizeof(char) * amino_seq_len, cudaMemcpyHostToDevice))
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

    /* CUDA Kerenl call */
    float kernel_time = 0.f;
    void *initialization_args[] = {&d_random_generator, &d_seed, &d_amino_seq_idx, &d_population, &d_obj_val, &d_obj_idx, &d_pql, &d_sorted_array};
    /* Even cycle args */
    void *even_mutation_args[] = {&d_random_generator, &d_amino_seq_idx, &d_tmp_population, &d_tmp_obj_val, &d_tmp_obj_idx, &d_tmp_pql, &d_population, &d_obj_val, &d_obj_idx, &d_pql, &d_sorted_array};
    void *even_sorting_args[] = {&d_tmp_obj_val, &d_sorted_array, &d_F_set, &d_Sp_set, &d_np, &d_rank_count};
    /* Odd cycle args */
    void *odd_mutation_args[] = {&d_random_generator, &d_amino_seq_idx, &d_population, &d_obj_val, &d_obj_idx, &d_pql, &d_tmp_population, &d_tmp_obj_val, &d_tmp_obj_idx, &d_tmp_pql, &d_sorted_array};
    void *odd_sorting_args[] = {&d_obj_val, &d_sorted_array, &d_F_set, &d_Sp_set, &d_np, &d_rank_count};


    // TODO : 마지막에 sorting 다음에 sorting 된 index 기반으로 가지고 오는거 해야 함
    CHECK_CUDA(cudaEventRecord(d_start))
    if (shared_vs_global)
    {
        CHECK_CUDA(cudaLaunchCooperativeKernel((void **)initializationKernel, initialization_blocks_num, initialization_threads_per_block, initialization_args, initialzation_shared_memory_size))
        for (int i = 0; i < gen_cycle_num; i++)
        {
            CHECK_CUDA(cudaMemset(d_F_set, false, sizeof(bool) * 2 * population_size * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_Sp_set, false, sizeof(bool) * 2 * population_size * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_rank_count, 0, sizeof(int) * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_np, 0, sizeof(int) * 2 * population_size))

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
    }
    else
    {
        CHECK_CUDA(cudaLaunchCooperativeKernel((void **)globalInitializationKernel, global_initialization_blocks_num, global_initialization_threads_per_block, initialization_args, global_initialzation_shared_memory_size))
        for (int i = 0; i < gen_cycle_num; i++)
        {
            CHECK_CUDA(cudaMemset(d_F_set, false, sizeof(bool) * 2 * population_size * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_Sp_set, false, sizeof(bool) * 2 * population_size * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_rank_count, 0, sizeof(int) * 2 * population_size))
            CHECK_CUDA(cudaMemset(d_np, 0, sizeof(int) * 2 * population_size))

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
    }
    CHECK_CUDA(cudaEventRecord(d_end))
    CHECK_CUDA(cudaEventSynchronize(d_end))
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time, d_start, d_end))
    CHECK_CUDA(cudaDeviceSynchronize())
    kernel_time /= 1000.f;
    printf("Kernel time : %f\n", kernel_time);

    /* Memory copy Device to Host */
    CHECK_CUDA(cudaMemcpy(h_population, d_population, sizeof(char) * solution_len * population_size * 2, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(h_obj_val, d_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(h_obj_idx, d_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(h_rank_count, d_rank_count, sizeof(int) * population_size * 2, cudaMemcpyDeviceToHost)) // 이것도 나중에 제거 부분

    for(int i=0;i<OBJECTIVE_NUM;i++)
    {
        for(int j=0;j<2;j++)
        {
            printf("%f ",ideal_nadir_array[i][j]);
        }
        printf("\n");
    }

#if 0
    /* Print */
    for (int i = 0; i < population_size; i++)
    {
        for (int j = 0; j < cds_num; j++)
        {
            for (int k = 0; k < cds_len; k++)
            {
                printf("%c", h_population[i * solution_len + cds_len * j + k]);
            }
            printf("\n");
        }
        printf("\n %d mCAI : %f\n", h_obj_idx[i * OBJECTIVE_NUM * 2 + MIN_CAI_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MIN_CAI_IDX]);
        printf("\n %d mCBP : %f\n", h_obj_idx[i * OBJECTIVE_NUM * 2 + MIN_CBP_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MIN_CBP_IDX]);
        printf("\n %d mHSC : %f\n", h_obj_idx[i * OBJECTIVE_NUM * 2 + MIN_HSC_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MIN_HSC_IDX]);
        printf("\n %d  %d mHD : %f\n", h_obj_idx[i * OBJECTIVE_NUM * 2 + MIN_HD_IDX * 2], h_obj_idx[i * OBJECTIVE_NUM * 2 + MIN_HD_IDX * 2 + 1], h_obj_val[i * OBJECTIVE_NUM + MIN_HD_IDX]);
        printf("\n %d MGC : %f\n", h_obj_idx[i * OBJECTIVE_NUM * 2 + MAX_GC_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MAX_GC_IDX]);
        printf("\n %d MSL : %f\n", h_obj_idx[i * OBJECTIVE_NUM * 2 + MAX_SL_IDX * 2], h_obj_val[i * OBJECTIVE_NUM + MAX_SL_IDX]);
        printf("\n");
    }

    printf("\n");
    for (int i = 0; i < 2 * population_size; i++)
    {
        if (h_rank_count[i] == 0)
        {
            break;
        }

        printf("%d rank   count : %d\n", i, h_rank_count[i]);
    }
    /* ------------------------------- end print --------------------------- */
#endif
 
    /* free host memory */
    free(amino_seq);
    free(h_amino_seq_idx);
    free(h_population);
    free(h_obj_val);
    free(h_obj_idx); // 나중에 제거
    free(h_reference_points);
    free(h_rank_count); // 나중에 제거

    /* free deivce memory */
    CHECK_CUDA(cudaEventDestroy(d_start))
    CHECK_CUDA(cudaEventDestroy(d_end))
    CHECK_CUDA(cudaFree(d_random_generator))
    CHECK_CUDA(cudaFree(d_seed))
    CHECK_CUDA(cudaFree(d_amino_seq_idx))
    CHECK_CUDA(cudaFree(d_population))
    CHECK_CUDA(cudaFree(d_obj_val))
    CHECK_CUDA(cudaFree(d_obj_idx))
    CHECK_CUDA(cudaFree(d_pql))

    CHECK_CUDA(cudaFree(d_tmp_population))
    CHECK_CUDA(cudaFree(d_tmp_obj_val))
    CHECK_CUDA(cudaFree(d_tmp_obj_idx))
    CHECK_CUDA(cudaFree(d_tmp_pql))
    CHECK_CUDA(cudaFree(d_sorted_array))
    CHECK_CUDA(cudaFree(d_np))
    CHECK_CUDA(cudaFree(d_F_set))
    CHECK_CUDA(cudaFree(d_Sp_set))
    CHECK_CUDA(cudaFree(d_rank_count))
    CHECK_CUDA(cudaFree(d_sol_struct))  // crowding distance sorting 안하면 제거해야하는 부분


    return EXIT_SUCCESS;
}
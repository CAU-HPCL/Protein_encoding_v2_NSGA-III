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

__global__ void mainKernel(curandStateXORWOW *random_generator, unsigned long long seed, const char *d_amino_seq_idx, char *d_population, float *d_obj_val, char *d_obj_idx, int *d_pql, char *d_tmp_population, float *d_tmp_obj_val, char *d_tmp_obj_idx, int *d_tmp_pql, int *d_sorted_array, bool *F_set, bool *Sp_set, int *d_np, int *d_rank_count)
{
    auto g = this_grid();
    auto tb = this_thread_block();
    curand_init(seed, g.thread_rank(), 0, &random_generator[g.thread_rank()]);

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

    int partition_num;
    curandStateXORWOW local_generator = random_generator[g.thread_rank()];
    partition_num = (c_amino_seq_len % tb.size() == 0) ? (c_amino_seq_len / tb.size()) : (c_amino_seq_len / tb.size()) + 1;
    for (int i = 0; i < partition_num; i++)
    {
        int idx = tb.size() * i + tb.thread_rank();
        if (idx < c_amino_seq_len)
        {
            s_amino_seq_idx[idx] = d_amino_seq_idx[idx];
        }
    }
    tb.sync();

    int cycle_partition_num;
    int i, j;
    cycle_partition_num = (c_N % g.num_blocks() == 0) ? (c_N / g.num_blocks()) : (c_N / g.num_blocks()) + 1;

    /* Solutions initialization */
    for (i = 0; i < cycle_partition_num; i++)
    {
        int solution_idx = g.num_blocks() * i + g.block_rank();
        if (solution_idx < c_N)
        {
            if (solution_idx == (c_N - 1))
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

            copySolution(tb, s_solution, s_obj_val, s_obj_idx, s_pql, &d_population[c_solution_len * solution_idx], &d_obj_val[OBJECTIVE_NUM * solution_idx], &d_obj_idx[OBJECTIVE_NUM * 2 * solution_idx], &d_pql[3 * solution_idx]);  
            copySolution(tb, s_solution, s_obj_val, s_obj_idx, s_pql, &d_tmp_population[c_solution_len * solution_idx], &d_tmp_obj_val[OBJECTIVE_NUM * solution_idx], &d_tmp_obj_idx[OBJECTIVE_NUM * 2 * solution_idx], &d_tmp_pql[3 * solution_idx]);
            d_sorted_array[solution_idx] = solution_idx;
        }
    }
    g.sync();

    /* Starting Generation */
    for (i = 0; i < c_gen_cycle_num; i++)
    {
        for (j = 0; j < cycle_partition_num; j++)
        {
            int solution_idx = g.num_blocks() * j + g.block_rank();
            if (solution_idx < c_N)
            {
                if (i != 0)
                {
                    copySolution(tb, &d_tmp_population[c_solution_len * d_sorted_array[solution_idx]], &d_tmp_obj_val[OBJECTIVE_NUM * d_sorted_array[solution_idx]], &d_tmp_obj_idx[OBJECTIVE_NUM * 2 * d_sorted_array[solution_idx]], &d_tmp_pql[3 * solution_idx], &d_population[c_solution_len * solution_idx], &d_obj_val[OBJECTIVE_NUM * solution_idx], &d_obj_idx[OBJECTIVE_NUM * 2 * solution_idx], &d_pql[3 * solution_idx]);
                    copySolution(tb, &d_population[c_solution_len * solution_idx], &d_obj_val[OBJECTIVE_NUM * solution_idx], &d_obj_idx[OBJECTIVE_NUM * 2 * solution_idx], &d_pql[3 * solution_idx], s_solution, s_obj_val, s_obj_idx, s_pql);
                }

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
                    mutationCBP(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_idx, SELECT_UPPER_RANDOM, i % 2);
                    break;
                case 3:
                    mutationHSC(tb, &local_generator, s_solution, s_amino_seq_idx, s_obj_idx, SELECT_UPPER_RANDOM, i % 2);
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

                copySolution(tb, s_solution, s_obj_val, s_obj_idx, s_pql, &d_population[c_solution_len * (c_N + solution_idx)], &d_obj_val[OBJECTIVE_NUM * (c_N + solution_idx)], &d_obj_idx[OBJECTIVE_NUM * 2 * (c_N + solution_idx)], &d_pql[3 * (c_N + solution_idx)]);
            }
        }
        g.sync();

        int copy_partition_num = ((c_N * 2) % g.num_blocks() == 0) ? ((c_N * 2) / g.num_blocks()) : ((c_N * 2) / g.num_blocks()) + 1;
        for (j = 0; j < copy_partition_num; j++)
        {
            int solution_idx = g.num_blocks() * j + g.block_rank();
            if (solution_idx < (c_N * 2))
            {
                copySolution(tb, &d_population[c_solution_len * solution_idx], &d_obj_val[OBJECTIVE_NUM * solution_idx], &d_obj_idx[OBJECTIVE_NUM * 2 * solution_idx], &d_pql[3 * solution_idx], &d_tmp_population[c_solution_len * solution_idx], &d_tmp_obj_val[OBJECTIVE_NUM * solution_idx], &d_tmp_obj_idx[OBJECTIVE_NUM * 2 * solution_idx], &d_tmp_pql[3 * solution_idx]);
            }
        }
        g.sync();

        /* Sorting */
        nonDominatedSorting(g, d_obj_val, d_sorted_array, F_set, Sp_set, d_np, d_rank_count);
    }

    /* Memory copy from shared memory to global memory */
    for (i = 0; i < cycle_partition_num; i++)
    {
        int solution_idx = g.num_blocks() * i + g.block_rank();
        if (solution_idx < c_N)
        {
            copySolution(tb, &d_tmp_population[c_solution_len * d_sorted_array[solution_idx]], &d_tmp_obj_val[OBJECTIVE_NUM * d_sorted_array[solution_idx]], &d_tmp_obj_idx[OBJECTIVE_NUM * 2 * d_sorted_array[solution_idx]], &d_tmp_pql[3 * d_sorted_array[solution_idx]], &d_population[c_solution_len * solution_idx], &d_obj_val[OBJECTIVE_NUM * solution_idx], &d_obj_idx[OBJECTIVE_NUM * 2 * solution_idx], &d_pql[3 * solution_idx]);
        }
    }

    return;
}

/*
argv[1] : Input file name
argv[2] : Population size (N)
argv[3] : Generation count (G)
argv[4] : Number of CDS
argv[5] : Mutation probability (Pm)
argv[6] : Number of threads per block

For example
../Protein_FASTA/Q5VZP5.fasta.txt  10 10 2 0.5 32
*/
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
    int threads_per_block = atoi(argv[6]);
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
    char *h_obj_idx; // 나중에 제거
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
    getReferencePoints(h_reference_points, OBJECTIVE_NUM, population_size);

    int blocks_num;
    int numBlocksPerSm = 0;
    size_t using_shared_memory_size = sizeof(float) * (OBJECTIVE_NUM + threads_per_block) + sizeof(char) * (amino_seq_len + solution_len + (OBJECTIVE_NUM * 2) + 1) + sizeof(int) * 6;
    size_t using_constant_memory_size = sizeof(codons_start_idx) + sizeof(syn_codons_num) + sizeof(codons) + sizeof(codons_weight) + sizeof(cps) + sizeof(int) * 5 + sizeof(char) + sizeof(float);

    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, mainKernel, threads_per_block, using_shared_memory_size))
    if (population_size > (deviceProp.multiProcessorCount * numBlocksPerSm))
    {
        blocks_num = (deviceProp.multiProcessorCount * numBlocksPerSm);
    }
    else
    {
        blocks_num = population_size;
    }
    // TODO: 나중에 다시 계산 필요한 부분
    size_t using_global_memory_size = sizeof(curandStateXORWOW) * (blocks_num * threads_per_block) + sizeof(unsigned long long) + sizeof(char) * (amino_seq_len + solution_len * population_size * 2 + OBJECTIVE_NUM * 2 * population_size * 2) + sizeof(float) * (OBJECTIVE_NUM * population_size * 2);

    /* Host Memory allocation */
    h_population = (char *)malloc(sizeof(char) * solution_len * population_size * 2);
    h_obj_val = (float *)malloc(sizeof(float) * OBJECTIVE_NUM * population_size * 2);
    h_obj_idx = (char *)malloc(sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2); // 나중에 제거

    /* Device Memory allocation */
    CHECK_CUDA(cudaEventCreate(&d_start))
    CHECK_CUDA(cudaEventCreate(&d_end))
    CHECK_CUDA(cudaMalloc((void **)&d_random_generator, sizeof(curandStateXORWOW) * blocks_num * threads_per_block))
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
    CHECK_CUDA(cudaMemcpyToSymbol(c_gen_cycle_num, &gen_cycle_num, sizeof(int)))

    /* CUDA Kerenl call */
    printf("Global memory usage : %lu bytes\n", using_global_memory_size);
    printf("Shared memory usage : %lu bytes\n", using_shared_memory_size);
    printf("Constant memory usage : %lu bytes\n", using_constant_memory_size);
    float kernel_time = 0.f;
    void *args[] = {&d_random_generator, &d_seed, &d_amino_seq_idx, &d_population, &d_obj_val, &d_obj_idx, &d_pql, &d_tmp_population, &d_tmp_obj_val, &d_tmp_obj_idx, &d_tmp_pql, &d_sorted_array, &d_F_set, &d_Sp_set, &d_np, &d_rank_count};
    CHECK_CUDA(cudaEventRecord(d_start))
    CHECK_CUDA(cudaLaunchCooperativeKernel((void **)mainKernel, blocks_num, threads_per_block, args, using_shared_memory_size))
    CHECK_CUDA(cudaDeviceSynchronize())
    CHECK_CUDA(cudaEventRecord(d_end))
    CHECK_CUDA(cudaEventSynchronize(d_end))
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time, d_start, d_end))
    kernel_time /= 1000.f;
    printf("Kernel time : %f\n", kernel_time);

    /* Memory copy Device to Host */
    CHECK_CUDA(cudaMemcpy(h_population, d_population, sizeof(char) * solution_len * population_size * 2, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(h_obj_val, d_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(h_obj_idx, d_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2, cudaMemcpyDeviceToHost))

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

    /* free host memory */
    free(amino_seq);
    free(h_amino_seq_idx);
    free(h_population);
    free(h_obj_val);
    free(h_obj_idx); // 나중에 제거
    free(h_reference_points);

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

    return EXIT_SUCCESS;
}
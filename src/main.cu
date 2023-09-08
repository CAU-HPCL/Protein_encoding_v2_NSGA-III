#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cooperative_groups.h>

#include "../include/common.cuh"
#include "../include/info.cuh"
#include "../include/mutation.cuh"
#include "../include/sorting.cuh"
#include "../include/utils.cuh"

#define _CRT_SECURE_NO_WARINGS

using namespace cooperative_groups;

__global__ void mainKernel(curandStateXORWOW *random_generator, unsigned long long seed, const char *d_amino_seq_idx, char *d_population, float *d_obj_val, char *d_obj_idx)
{
    /* TODO : population 크기가 블럭 수 보다 많으면 체크해서 반복 하는 코드 작성 필요 */
    auto g = this_grid();
    auto tb = this_thread_block();

    curand_init(seed, g.thread_rank(), 0, &random_generator[g.thread_rank()]); // Random number generator initialization in global memory

    /* Shared memory allocation */
    extern __shared__ int smem[];
    __shared__ float *s_obj_val;
    __shared__ float *s_obj_buffer;
    __shared__ char *s_amino_seq_idx;
    __shared__ char *s_solution;
    __shared__ char *s_obj_idx;

    s_obj_val = (float *)smem;
    s_obj_buffer = (float *)&s_obj_val[OBJECTIVE_NUM];
    s_amino_seq_idx = (char *)&s_obj_buffer[tb.size()];
    s_solution = (char *)&s_amino_seq_idx[c_amino_seq_len];
    s_obj_idx = (char *)&s_solution[c_cds_len * c_cds_num];

    /* Variable initialization */
    curandStateXORWOW local_generator = random_generator[g.thread_rank()];

    /* Solutions initialization + (Calculating objective function) */
    genPopulation(tb, &local_generator, s_amino_seq_idx, s_solution, s_obj_val, s_obj_idx, RANDOM_GEN, s_obj_buffer);

    /* Mutation */

    /* Calculating objective function */

    /* Sorting */


    /* Memory copy from shared memory to global memory */

    return;
}

/*
argv[0] : Input file name
argv[1] : Population size (N)
argv[2] : Generation count (G)
argv[3] : Number of CDS
argv[4] : Mutation probability (Pm)
argv[5] : Number of threads per block
*/
int main(const int argc, const char *argv[])
{
    srand((unsigned int)time(NULL)); // for random generator initialization

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
    int population_size = atoi(argv[1]);
    int gen_cycle_num = atoi(argv[2]);
    int cds_num = atoi(argv[3]);
    float mutation_prob = atof(argv[4]);
    int threads_per_block = atoi(argv[5]);
    if ((population_size <= 0) || (gen_cycle_num < 0) || (cds_num <= 0) || (mutation_prob < 0.f) || (mutation_prob > 1.f))
    {
        printf("Line : %d Please cheking input parameters.. \n", __LINE__);
        return EXIT_FAILURE;
    }

    /* Preprocessing */
    FILE *fp;
    char buffer[256];
    char *amino_seq; // store amino sequences
    int amino_seq_len, cds_len, solution_len;
    int idx;

    fp = fopen(argv[0], "r");
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
    amino_seq[idx] = NULL;
    amino_seq_len = idx;
    cds_len = amino_seq_len * CODON_SIZE;
    solution_len = cds_len * cds_num;
    fclose(fp);

    unsigned long long seed = (unsigned long long)rand();
    char *h_amino_seq_idx;
    char *h_codons_start_idx;
    char *h_population;
    float *h_obj_val;
    // char *h_obj_idx;

    curandStateXORWOW *d_random_generator;
    cudaEvent_t d_start, d_end;
    unsigned long long *d_seed;
    char *d_amino_seq_idx;
    char *d_population;
    float *d_obj_val;
    char *d_obj_idx;

    h_amino_seq_idx = (char *)malloc(sizeof(char) * amino_seq_len);
    for (int i = 0; i < amino_seq_len; i++)
    {
        h_amino_seq_idx[i] = findAminoIndex(amino_seq[i]);
    }
    h_codons_start_idx = (char *)malloc(sizeof(char) * 21);
    h_codons_start_idx[0] = 0;
    for (int i = 1; i < 21; i++)
    {
        h_codons_start_idx[i] = h_codons_start_idx[i - 1] + syn_codons_num[i - 1];
    }

    int blocks_num; // 2N 보다 작은 경우 interate 하게 바꾸어 주어야 하는 처리 필요
    int numBlocksPerSm = 0;
    size_t using_global_memory_size = 0;
    size_t using_shared_memory_size = 0;
    size_t using_constant_memory_size = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, mainKernel, threads_per_block, using_shared_memory_size))

    if ((deviceProp.multiProcessorCount * numBlocksPerSm) > population_size)
    {
        blocks_num = deviceProp.multiProcessorCount * numBlocksPerSm;
    }
    else
    {
        blocks_num = population_size;
    }

    printf("Global memory usage : \n");
    printf("Shared memory usage : \n");
    printf("Constant memory usage : \n");

    /* Host Memory allocation */
    h_population = (char *)malloc(sizeof(char) * solution_len * population_size * 2);
    h_obj_val = (float *)malloc(sizeof(float) * OBJECTIVE_NUM * population_size * 2);

    /* Device Memory allocation */
    CHECK_CUDA(cudaMalloc((void **)&d_random_generator, sizeof(curandStateXORWOW) * blocks_num * threads_per_block))
    CHECK_CUDA(cudaEventCreate(&d_start))
    CHECK_CUDA(cudaEventCreate(&d_end))
    CHECK_CUDA(cudaMalloc((void **)&d_seed, sizeof(unsigned long long)))
    CHECK_CUDA(cudaMalloc((void **)&d_amino_seq_idx, sizeof(char) * amino_seq_len))
    CHECK_CUDA(cudaMalloc((void **)&d_population, sizeof(char) * solution_len * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2))
    CHECK_CUDA(cudaMalloc((void **)&d_obj_idx, sizeof(char) * OBJECTIVE_NUM * 2 * population_size * 2)) // mHD cds idx is 2

    /* Memory copy Host to Device */
    CHECK_CUDA(cudaMemcpy(d_seed, &seed, sizeof(unsigned long long), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_amino_seq_idx, h_amino_seq_idx, sizeof(char) * amino_seq_len, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpyToSymbol(c_codons_start_idx, h_codons_start_idx, sizeof(char) * 21))
    CHECK_CUDA(cudaMemcpyToSymbol(c_syn_codons_num, syn_codons_num, sizeof(syn_codons_num)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_codons, codons, sizeof(codons)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_codons_weight, codons_weight, sizeof(codons_weight)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_cps, cps, sizeof(cps)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_population_size, &population_size, sizeof(int)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_amino_seq_len, &amino_seq_len, sizeof(int)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_cds_len, &cds_len, sizeof(int)))
    CHECK_CUDA(cudaMemcpyToSymbol(c_cds_num, &cds_num, sizeof(int)))

    /* CUDA Kerenl call */
    float kernel_time = 0.f;
    void *args[] = {&d_random_generator, &d_seed, &d_amino_seq_idx, &d_population, &d_obj_val, &d_obj_idx};
    CHECK_CUDA(cudaEventRecord(d_start))
    CHECK_CUDA(cudaLaunchCooperativeKernel((void **)mainKernel, blocks_num, threads_per_block, args))
    CHECK_CUDA(cudaEventRecord(d_end))
    CHECK_CUDA(cudaEventSynchronize(d_end))
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time, d_start, d_end))
    kernel_time /= 1000.f;

    /* Memory copy Device to Host */
    CHECK_CUDA(cudaMemcpy(h_population, d_population, sizeof(char) * solution_len * population_size * 2, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(h_obj_val, d_obj_val, sizeof(float) * OBJECTIVE_NUM * population_size * 2, cudaMemcpyDeviceToHost))

    /* free host memory */
    free(amino_seq);
    free(h_amino_seq_idx);
    free(h_codons_start_idx);
    free(h_population);
    free(h_obj_val);

    /* free deivce memory */
    CHECK_CUDA(cudaFree(d_seed))
    CHECK_CUDA(cudaFree(d_random_generator))
    CHECK_CUDA(cudaEventDestroy(d_start))
    CHECK_CUDA(cudaEventDestroy(d_end))
    CHECK_CUDA(cudaFree(d_amino_seq_idx))
    CHECK_CUDA(cudaFree(d_population))
    CHECK_CUDA(cudaFree(d_obj_val))
    CHECK_CUDA(cudaFree(d_obj_idx))

    return EXIT_SUCCESS;
}
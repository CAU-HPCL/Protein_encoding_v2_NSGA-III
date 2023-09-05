/*
1. Mutation functions of Objectvie functions
    1.1 Select the best synonymous codon
    1.2 Select random codon among synonymous codons better than current codon
    1.3 Select random codon amsmon synonymous codon
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "common.cuh"
#include "utils.cuh"

#define SELECT_UPPER 0
#define SELECT_UPPER_RANDOM 1
#define SELECT_RANDOM 2

__device__ void mutationRandom(const int mutation_type, const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const int aminoacid_idx)
{
    return;
}

__device__ void mutationCAI(const int mutation_type, const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const int aminoacid_idx)
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
            solution[solution_idx] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE];
            solution[solution_idx + 1] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE + 1];
            solution[solution_idx + 2] = c_codons[(c_codons_start_idx[aminoacid_idx] + c_syn_codons_num[aminoacid_idx] - 1) * CODON_SIZE + 2];
            break;

        case SELECT_UPPER_RANDOM:
            if (cur_codon_idx != c_syn_codons_num[aminoacid_idx] - 1)
            {
                new_codon_idx = (char)(curand_uniform(random_generator) * (c_syn_codons_num[aminoacid_idx] - 1 - cur_codon_idx));
                while (new_codon_idx == (c_syn_codons_num[aminoacid_idx] - 1- cur_codon_idx))
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
    }

    return;
}

__device__ void mutationCBP(const int mutation_type, const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const int aminoacid_idx)
{
    return;
}
__device__ void mutationHD(const int mutation_type, const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const int aminoacid_idx)
{
    return;
}

__device__ void mutationHSC(const int mutation_type, const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const int aminoacid_idx)
{
    return;
}

__device__ void mutationGC(const int mutation_type, const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const int aminoacid_idx)
{
    return;
}

__device__ void mutationSL(const int mutation_type, const float mutation_prob, curandStateXORWOW *random_generator, char *solution, const int solution_idx, const int aminoacid_idx)
{
    return;
}
#ifndef UTILS_H
#define UTILS_H

__device__ char findIndexAmongSynonymousCodons(const char *cur_codon, const char *syn_codons, const char syn_codons_num);

__device__ char countBothSidesHSC(const char *left_codon, const char *cur_codon, const char *right_codon);


#endif
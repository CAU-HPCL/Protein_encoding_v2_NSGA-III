/*
Information of the codons & codon pairs
*/

#ifndef INFO_H
#define INFO_H

#include "utils.cuh"

const char aminoacids[21] = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z'}; // The last means stop
const char codons_start_idx[21] = {0, 4, 6};
const char syn_codons_num[21] = {4, 2, 2, 2, 2, 4, 2, 3, 2, 6, 1, 2, 4, 2, 6, 6, 4, 4, 1, 2, 3};
const char codons[TOTAL_CODON_NUM * CODON_SIZE + 1] = "GCGGCAGCCGCU\
UGCUGU\
GACGAU\
GAGGAA\
UUUUUC\
GGGGGAGGCGGU\
CACCAU\
AUAAUCAUU\
AAAAAG\
CUCCUGCUUCUAUUAUUG\
AUG\
AAUAAC\
CCGCCCCCUCCA\
CAGCAA\
CGGCGACGCAGGCGUAGA\
UCGAGCAGUUCAUCCUCU\
ACGACAACCACU\
GUAGUGGUCGUU\
UGG\
UAUUAC\
UAGUGAUAA";

const float codons_weight[TOTAL_CODON_NUM] = {

};

const float cps[(TOTAL_CODON_NUM - 3) * (TOTAL_CODON_NUM - 3)] = {

};


#endif
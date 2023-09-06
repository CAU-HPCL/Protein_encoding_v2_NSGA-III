/*
Information of the codons & codon pairs
*/

#ifndef INFO_H
#define INFO_H

#include "../include/common.cuh"

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

/*
아미노산 AB 면
아미노산 A 에 해당 하는 코돈 중 cur index 찾아서 index 값이 a * 61 쪽
B 에 해당 하는 코돈 cur index 가 b  쪽이 될 것 인데..
cps 오름차순 정렬이 안되는데,,
그러면 왼쪽 인덱스랑 오른쪽 인덱스로 cps 값을 찾는 것이 가능하긴 함
코돈 1개당 최대 가능 한 조합이 맥스로 왼쪽 6개
오른 쪽 6개
현재 값보다 높은걸 선택하게 해야하니까
*/
const float cps[(TOTAL_CODON_NUM - STOP_CODON_NUM) * (TOTAL_CODON_NUM - STOP_CODON_NUM)] = {

};


#endif
#ifndef SORTING_H
#define SORTING_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "../include/common.cuh"

using namespace cooperative_groups;


/*
1. non-dominated sorting
2. reference direction sorting
    - reference points
    - 각 objective 값 정규화 하기
        현재까지 가장 좋지 않은 것 저장하기
        현재까지 가장 좋은 것 저장하기
        hyperplane 만들어 지는지 확인하기
        만들어진 hyperplane 의 절편 값 체크하기
    - reference point 에 associated 된 solution 과 거리와 해당 점에서 N에 포함된 solution 개수
    - 하나씩 랜덤하게 뽑느다.
        하나도 없으면 가장 작은 것을 뽑도록 하기
        하나라도 있으면 랜덤하게 뽑기
        뽑히면 더 이상 고려되지 않도록 하기
*/
/* 각 블럭이 담당한 solution 이 존재
하나의 블럭은 본인이 담당한 solution 과 가장 가까운 reference point 를 계산해야함
그러면 하나의 쓰레드는 하나의 수직선과 거리를 계산하게 됨
그러면 수직 거리 계산하는 것은 objective function 값과 reference point 만 있으면 되긴 함
근데 각 수직선에 몇개의 solution 이 할당 될 지 모르기 때문에 이게 함정인데, 어떻게 처리할 서
*/

#endif
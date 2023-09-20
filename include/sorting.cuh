#ifndef SORTING_H
#define SORTING_H

#include <stdio.h>
#include <string.h>
#include <python3.10/Python.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "../include/common.cuh"

using namespace cooperative_groups;

#define _CRT_SECURE_NO_WARNINGS

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

__device__ int rank_count = 0; // reference direction sorting 에 포함될 solution 개수 저장할 변수
__device__ int cur_front = 0;

__host__ void getReferencePoints(float *const h_reference_points, const int obj_num, const int ref_num)
{
    FILE *fp;
    char buffer[128];
    char py_command[128] = "python3 ./reference_points.py";
    char tmp_str1[64];
    char tmp_str2[64];
    int i, j;

    sprintf(tmp_str1, " %d", obj_num);
    sprintf(tmp_str2, " %d", ref_num);
    strcat(py_command, tmp_str1);
    strcat(py_command, tmp_str2);

    fp = popen(py_command, "r");
    if (fp == NULL)
    {
        perror("Failed to execute Python script. \n");
        return;
    }

    i = 0;
    j = 0;
    while (fgets(buffer, sizeof(buffer), fp) != NULL)
    {
        h_reference_points[i * OBJECTIVE_NUM + j] = atof(buffer);
        if (++j == OBJECTIVE_NUM)
        {
            i += 1;
            j = 0;
        }
    }
    pclose(fp);

    return;
}

__device__ bool paretoComparison(const float *new_obj_val, const float *old_obj_val)
{
    if ((new_obj_val[MIN_CAI_IDX] == old_obj_val[MIN_CAI_IDX]) && // weak pareto dominance
        (new_obj_val[MIN_HD_IDX] == old_obj_val[MIN_HD_IDX]) &&
        (new_obj_val[MIN_CBP_IDX] == old_obj_val[MIN_CBP_IDX]) &&
        (new_obj_val[MIN_HSC_IDX] == old_obj_val[MIN_HSC_IDX]) &&
        (new_obj_val[MAX_GC_IDX] == old_obj_val[MAX_GC_IDX]) &&
        (new_obj_val[MAX_SL_IDX] == old_obj_val[MAX_SL_IDX]))
        return false;
    else if ((new_obj_val[MIN_CAI_IDX] >= old_obj_val[MIN_CAI_IDX]) &&
             (new_obj_val[MIN_HD_IDX] >= old_obj_val[MIN_HD_IDX]) &&
             (new_obj_val[MIN_CBP_IDX] >= old_obj_val[MIN_CBP_IDX]) &&
             (new_obj_val[MIN_HSC_IDX] >= old_obj_val[MIN_HSC_IDX]) &&
             (new_obj_val[MAX_GC_IDX] <= old_obj_val[MAX_GC_IDX]) &&
             (new_obj_val[MAX_SL_IDX] <= old_obj_val[MAX_SL_IDX]))
        return true;
    else
        return false;
}

// 결과로 sorting 완료된 d_sorted_array solution index 를 반환하고, 추가적으로 rank_count, cur_fornt, d_rank_count 가 업데이트 된다.
__device__ void nonDominatedSorting(grid_group g, float *d_obj_val, int *d_sorted_array, bool *F_set, bool *Sp_set, int *d_np, int *d_rank_count)
{
    int i, j, k;
    int idx;
    int r_2N = c_N * 2;

    int g_tid;
    int cycle_partition_num;

    cycle_partition_num = (r_2N % g.size() == 0) ? (r_2N / g.size()) : (r_2N / g.size()) + 1;

    if (g.thread_rank() == 0)
    {
        rank_count = 0;
        cur_front = 0;
    }

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < r_2N)
        {
            d_rank_count[g_tid] = 0; // indicate number of solution in rank
            d_np[g_tid] = 0;         // indicate solution's dominated number of solution
        }
    }
    g.sync();

    /* -------------------- 1st front setting -------------------- */
    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < r_2N)
        {
            for (j = 0; j < r_2N; j++)
            {
                if (g_tid != j)
                {
                    if (paretoComparison(&d_obj_val[g_tid * OBJECTIVE_NUM], &d_obj_val[j * OBJECTIVE_NUM]))
                    {
                        Sp_set[g_tid * r_2N + j] = true;
                    }
                    else if (paretoComparison(&d_obj_val[j * OBJECTIVE_NUM], &d_obj_val[g_tid * OBJECTIVE_NUM]))
                    {
                        d_np[g_tid] += 1;
                    }
                }
            }

            if (d_np[g_tid] == 0)
            {
                F_set[g_tid] = true;
                idx = atomicAdd(&rank_count, 1); // atomicAdd return value is stored memory value before add operation
                d_sorted_array[idx] = g_tid;
                atomicAdd(&d_rank_count[cur_front], 1);
            }
        }
    }
    g.sync();

    if (rank_count < c_N) // 딱 N 개로 떨어질 때 체크하는 거 필요함 딱 N 개면 reference direction sorting 을 하지 않아도 되기 때문임
    {
        /* -------------------- After 1st front setting -------------------- */
        if (g.thread_rank() == 0)
        {
            cur_front += 1;
        }
        g.sync();

        for (i = 0; i < r_2N - 1; i++)
        {
            for (j = 0; j < cycle_partition_num; j++)
            {
                g_tid = g.size() * j + g.thread_rank();
                if (g_tid < r_2N)
                {
                    for (k = 0; k < r_2N; k++)
                    {
                        if (F_set[(cur_front - 1) * r_2N + k] && Sp_set[k * r_2N + g_tid])
                        {
                            d_np[g_tid] -= 1;
                            if (d_np[g_tid] == 0)
                            {
                                F_set[cur_front * r_2N + g_tid] = true;
                                idx = atomicAdd(&rank_count, 1);
                                d_sorted_array[idx] = g_tid;
                                atomicAdd(&d_rank_count[cur_front], 1);
                            }
                        }
                    }
                }
            }
            g.sync();

            if (rank_count >= c_N)
            {
                break;
            }

            if (g.thread_rank() == 0)
            {
                cur_front += 1;
            }
            g.sync();
        }
    }
}

#endif
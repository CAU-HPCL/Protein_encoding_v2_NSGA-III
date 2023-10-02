#ifndef SORTING_H
#define SORTING_H

#include <stdio.h>
#include <string.h>
#include <python3.10/Python.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "../include/common.cuh"
#include "../include/utils.cuh"

using namespace cooperative_groups;

#define _CRT_SECURE_NO_WARNINGS

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

__device__ int rank_count; // reference direction sorting 에 포함될 solution 개수 저장할 변수
__device__ int cur_front;
__device__ int sorting_idx;
__device__ bool N_cut_check;

__device__ void nonDominatedSorting(grid_group g, const float *d_obj_val, int *d_sorted_array, bool *F_set, bool *Sp_set, int *d_np, int *d_rank_count)
{
    /* N 크기 커졌을 때 오버플로 방지를 위함 */
    size_t i, j, k;
    size_t idx;
    size_t r_2N = c_N * 2;

    size_t g_tid;
    size_t cycle_partition_num;

    cycle_partition_num = (r_2N % g.size() == 0) ? (r_2N / g.size()) : (r_2N / g.size()) + 1;

    if (g.thread_rank() == 0)
    {
        rank_count = 0;
        cur_front = 0;
        sorting_idx = 0;
        N_cut_check = false;
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
                if ((rank_count == c_N) && (g.thread_rank() == 0))
                {
                    N_cut_check = true;
                }
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

/* 현재 objective 값 계산하고 atomic 으로 ideal 과 nadir 값을 업데이트 하도록 해놓았는데,
이 부분은 나중에 변이 후 쓰레드들이 divide & conquer 전략을 사용해서 ideal 과 nadir 값을 업데이트 하는 식으로 바꾸도록 함.
왜냐하면 동기화로 인한 시간 소요가 너무 큼
*/

typedef struct
{
    int sol_idx;
    float corwding_dist;
    float obj_val[OBJECTIVE_NUM];
} Sol;

__device__ void Sol_assign(Sol *s1, Sol *s2)
{
    int i;

    s1->corwding_dist = s2->corwding_dist;
    s1->sol_idx = s2->sol_idx;
    for (i = 0; i < OBJECTIVE_NUM; i++)
    {
        s1->obj_val[i] = s2->obj_val[i];
    }

    return;
}

__device__ void CompUp(Sol *s1, Sol *s2, int idx)
{
    Sol tmp;

    if (s1->obj_val[idx] > s2->obj_val[idx])
    {
        Sol_assign(&tmp, s1);
        Sol_assign(s1, s2);
        Sol_assign(s2, &tmp);
    }
    return;
}

__device__ void CompDownCrowd(Sol *s1, Sol *s2)
{
    Sol tmp;

    if (s1->corwding_dist < s2->corwding_dist)
    {
        Sol_assign(&tmp, s1);
        Sol_assign(s1, s2);
        Sol_assign(s2, &tmp);
    }

    return;
}

/* 정규화 식 : (Objective function 값 - ideal point 값) / (nadir point 값 - ideal point 값) */
__device__ void crowdingDistanceSorting(grid_group g, const float *d_obj_val, int *d_sorted_array, bool *F_set, int *d_rank_count, Sol *d_sol_struct)
{
    if (N_cut_check) // 딱 N개로 잘려있는 경우는 할 필요가 없기 때문에
    {
        return;
    }

    int i, j;
    int sol_idx;
    int sec1, sec2;
    int r_2N = c_N * 2;
    int cycle_partition_num = (r_2N % g.size() == 0) ? (r_2N / g.size()) : (r_2N / g.size()) + 1;
    int g_tid;

    sol_idx = 0;
    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < r_2N)
        {
            if (F_set[cur_front * r_2N + g_tid])    // sorting 에 사용되는 것을 값을 미리 정규화 해 주는 부분
            {
                sol_idx = atomicAdd(&sorting_idx, 1);
                d_sol_struct[sol_idx].sol_idx = g_tid;
                d_sol_struct[sol_idx].corwding_dist = 0.f;
                d_sol_struct[sol_idx].obj_val[MIN_CAI_IDX] = (d_obj_val[g_tid * OBJECTIVE_NUM + MIN_CAI_IDX] - ideal_nadir_array[MIN_CAI_IDX][0]) / (ideal_nadir_array[MIN_CAI_IDX][1] - ideal_nadir_array[MIN_CAI_IDX][0]);
                d_sol_struct[sol_idx].obj_val[MIN_CBP_IDX] = (d_obj_val[g_tid * OBJECTIVE_NUM + MIN_CBP_IDX] - ideal_nadir_array[MIN_CBP_IDX][0]) / (ideal_nadir_array[MIN_CBP_IDX][1] - ideal_nadir_array[MIN_CBP_IDX][0]);
                d_sol_struct[sol_idx].obj_val[MIN_HSC_IDX] = (d_obj_val[g_tid * OBJECTIVE_NUM + MIN_HSC_IDX] - ideal_nadir_array[MIN_HSC_IDX][0]) / (ideal_nadir_array[MIN_HSC_IDX][1] - ideal_nadir_array[MIN_HSC_IDX][0]);
                d_sol_struct[sol_idx].obj_val[MIN_HD_IDX] = (d_obj_val[g_tid * OBJECTIVE_NUM + MIN_HD_IDX] - ideal_nadir_array[MIN_HD_IDX][0]) / (ideal_nadir_array[MIN_HD_IDX][1] - ideal_nadir_array[MIN_HD_IDX][0]);
                d_sol_struct[sol_idx].obj_val[MAX_GC_IDX] = (d_obj_val[g_tid * OBJECTIVE_NUM + MAX_GC_IDX] - ideal_nadir_array[MAX_GC_IDX][0]) / (ideal_nadir_array[MAX_GC_IDX][1] - ideal_nadir_array[MAX_GC_IDX][0]);
                d_sol_struct[sol_idx].obj_val[MAX_SL_IDX] = (d_obj_val[g_tid * OBJECTIVE_NUM + MAX_SL_IDX] - ideal_nadir_array[MAX_SL_IDX][0]) / (ideal_nadir_array[MAX_SL_IDX][1] - ideal_nadir_array[MAX_SL_IDX][0]);
            }
        }
    }
    g.sync();

    for (i = 0; i < OBJECTIVE_NUM; i++)
    {
        // sorting objective function ascending order
        sec1 = 1;
        while (sec1 < d_rank_count[cur_front])
        {
            for (j = 0; j < cycle_partition_num; j++)
            {
                g_tid = g.size() * j + g.thread_rank();
                if ((g_tid % (sec1 * 2) < sec1) && ((sec1 * 2 * (g_tid / (sec1 * 2) + 1) - g_tid % (sec1 * 2) - 1) < d_rank_count[cur_front]))
                {
                    CompUp(&d_sol_struct[g_tid], &d_sol_struct[sec1 * 2 * (g_tid / (sec1 * 2) + 1) - (g_tid % (sec1 * 2)) - 1], i);
                }
            }
            sec2 = sec1 / 2;
            g.sync();

            while (sec2 != 0)
            {
                for (j = 0; j < cycle_partition_num; j++)
                {
                    g_tid = g.size() * j + g.thread_rank();
                    if ((g_tid % (sec2 * 2) < sec2) && (g_tid + sec2 < d_rank_count[cur_front]))
                    {
                        CompUp(&d_sol_struct[g_tid], &d_sol_struct[g_tid + sec2], i);
                    }
                }
                sec2 /= 2;
                g.sync();
            }

            sec1 *= 2;
        }
        g.sync();

        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < d_rank_count[cur_front])
            {
                if (g_tid == 0)
                {
                    d_sol_struct[g_tid].corwding_dist = 10000.f;
                }
                else if (g_tid == d_rank_count[cur_front] - 1)
                {
                    d_sol_struct[g_tid].corwding_dist = 10000.f;
                }
                else
                {
                    d_sol_struct[g_tid].corwding_dist += d_sol_struct[g_tid + 1].obj_val[i] - d_sol_struct[g_tid - 1].obj_val[i];
                }
            }
        }
        g.sync();
    }

    // sort crowding distance descending order
    sec1 = 1;
    while (sec1 < d_rank_count[cur_front])
    {
        for (i = 0; i < cycle_partition_num; i++)
        {
            g_tid = g.size() * i + g.thread_rank();
            if ((g_tid % (sec1 * 2)) < sec1 && ((sec1 * 2 * (g_tid / (sec1 * 2) + 1) - g_tid % (sec1 * 2) - 1) < d_rank_count[cur_front]))
            {
                CompDownCrowd(&d_sol_struct[g_tid], &d_sol_struct[sec1 * 2 * (g_tid / (sec1 * 2) + 1) - (g_tid % (sec1 * 2)) - 1]);
            }
        }
        sec2 = sec1 / 2;
        g.sync();

        while (sec2 != 0)
        {
            for (i = 0; i < cycle_partition_num; i++)
            {
                g_tid = g.size() * i + g.thread_rank();

                if ((g_tid % (sec2 * 2) < sec2) && (g_tid + sec2 < d_rank_count[cur_front]))
                {
                    CompDownCrowd(&d_sol_struct[g_tid], &d_sol_struct[g_tid + sec2]);
                }
            }
            sec2 /= 2;
            g.sync();
        }

        sec1 *= 2;
    }
    g.sync();

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();

        if (g_tid < d_rank_count[cur_front])
        {
            d_sorted_array[rank_count - d_rank_count[cur_front] + g_tid] = d_sol_struct[g_tid].sol_idx;
        }
    }
    return;
}



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

typedef struct
{
    float reference_point[OBJECTIVE_NUM];
    int *solution_idx;
    float *distance;
    int N_include_solution_num;
    int associate_solution_num;
} reference_point_struct;

#endif
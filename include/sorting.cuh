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

__device__ int rank_count; // reference direction sorting 에 포함될 solution 개수 저장할 변수
__device__ int cur_front;
__device__ int sorting_idx;
__device__ bool N_cut_check;
__device__ float true_ideal_value[OBJECTIVE_NUM]; // **** 여기는 hypervolume 을 구하거나 정규화시 빠르게 사용할 수 있는 부분이기 때문에 나중에 체크 필요 ****
__device__ float true_nadir_value[OBJECTIVE_NUM]; // **** 여기는 hypervolume 을 구하거나 정규화시 빠르게 사용할 수 있는 부분이기 때문에 나중에 체크 필요 ****

__device__ float estimated_ideal_value[OBJECTIVE_NUM] = {__FLT_MIN__, __FLT_MIN__, __FLT_MIN__, __FLT_MIN__, __FLT_MAX__, __FLT_MAX__};
__device__ float estimated_nadir_value[OBJECTIVE_NUM] = {__FLT_MAX__, __FLT_MAX__, __FLT_MAX__, __FLT_MAX__, __FLT_MIN__, __FLT_MIN__};


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

#if 0
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
            if (F_set[cur_front * r_2N + g_tid]) // sorting 에 사용되는 것을 값을 미리 정규화 해 주는 부분
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
#endif

typedef struct
{
    float reference_point[OBJECTIVE_NUM];
    int *solution_idx;
    float *distance;
    int N_include_solution_num;
    int associate_solution_num;
} reference_point_struct;

/*
--------------------------------------------------
정리
Overall Process
1. non-dominated sorting
2. reference points based sorting
    2.1 normalization
        2.1.1 Estimation of ideal value & nadir value
            - ideal 값은 non-dominated front(rank 0) 중에서 가장 좋은 값을 사용 -> generation 마다 업데이트 되야 함
            - nadir 값은 3가지 방법 존재
                - Maximum of Non-dominated Front(MNDF)
                    > non-dominated front 에서 가장 좋지 않은 값을 nadir 값으로 사용하는 방법
                    > 만약 하나의 solution 으로 인해 ideal 값과 nadir 값이 동일한 상황이 생길 수 있기 때문에 이럴 때는 rank 1 즉, 다음 랭크에서 값을 구하게 됨
                - Maximum of Extreme Points(ME)
                    > 가장 최신의 극점과, 2N 크기의 merged population 에서 objective 개수 만큼 극점을 구하는 방법
                    > 이 방법은, ASF funtion 을 사용했을 때, 각 objective 에 대해 가장 작은 solution 의 값의 해당 obejctive 값을 총합해서 nadir 값을 구하게 됨
                - Revised Hyperplane through Extreme Points(HYP)
                    > 극점의 nadir 값에서 ideal 값을 빼고 해당 점들을 지나는 hyperpalen 의 각 objective 에 대한 intercept 를 nadir 값으로 사용하는 방법
                    > 이 방법은 intercept 가 음수 일 때와, hyper plane 이 형성되지 않을 때에 대한 예외적 처리가 필요
                    > 형성되지 않으면 MNDF 방법을 쓰게 됨
    2.2 association
    2.3 niching

    update_extreme_points
    find_hyperplane
    find_intercepts

- 정규화시킨 값을 어떻게 저장할 건지?

- 점들에 대해서 어떻게 처리할 건지?

NSGAII 도 정규화에 대한 정확한 방법이 없기 때문에 NSGAIII 와 비교시 같은 정규화된 값을 가지고 crowding distance sorting 한 것으로 해야할 것 같음
또한, 이미 ideal point 값과 nadir point 값을 알고 있다면 해당 값을 사용해서 정규화를 해도 되는 것은 확인된 부분임
*/


// 2N 크기의 버퍼에서 최솟값 찾음.
__device__ float findMinValue(grid_group g, float *buffer)
{
    int cycle_partition_num;
    int g_tid;
    int i, j;

    i = c_N;
    while (true)
    {
        cycle_partition_num = (i % g.size() == 0) ? (i / g.size()) : (i / g.size()) + 1;
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if ((g_tid < i) && (buffer[g_tid + i] < buffer[g_tid]))
            {
                buffer[g_tid] = buffer[g_tid + i];
            }
        }
        g.sync();

        if (i == 1)
        {
            break;
        }

        if ((i % 2 == 1) && (g.thread_rank() == 0))
        {
            if (buffer[i - 1] < buffer[0])
            {
                buffer[0] = buffer[i - 1];
            }
        }
        g.sync();

        i /= 2;
    }

    return buffer[0];
}

// 2N 크기의 버퍼에서 최댓값 찾음.
__device__ float findMaxValue(grid_group g, float *buffer)
{
    int cycle_partition_num;
    int g_tid;
    int i, j;

    i = c_N;
    while (true)
    {
        cycle_partition_num = (i % g.size() == 0) ? (i / g.size()) : (i / g.size()) + 1;
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if ((g_tid < i) && (buffer[g_tid + i] > buffer[g_tid]))
            {
                buffer[g_tid] = buffer[g_tid + i];
            }
        }
        g.sync();

        if (i == 1)
        {
            break;
        }

        if ((i % 2 == 1) && (g.thread_rank() == 0))
        {
            if (buffer[i - 1] > buffer[0])
            {
                buffer[0] = buffer[i - 1];
            }
        }
        g.sync();

        i /= 2;
    }

    return buffer[0];
}

// 매 generation 마다 estimated ideal value 업데이트 필요함.
__device__ void updateIdealValue(grid_group g, const float *obj_val, float *buffer, const int *d_sorted_array, const int *d_rank_count)
{
    int cycle_partition_num;
    float min, max;
    int g_tid;
    int i;

    cycle_partition_num = ((c_N * 2) % g.size() == 0) ? ((c_N * 2) / g.size()) : ((c_N * 2) / g.size()) + 1;

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MIN_CAI_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MIN__;
            }
        }
    }
    g.sync();
    max = findMaxValue(g, buffer);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MIN_CAI_IDX] = max;
    }

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MIN_CBP_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MIN__;
            }
        }
    }
    g.sync();
    max = findMaxValue(g, buffer);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MIN_CBP_IDX] = max;
    }

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MIN_HSC_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MIN__;
            }
        }
    }
    g.sync();
    max = findMaxValue(g, buffer);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MIN_HSC_IDX] = max;
    }

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MIN_HD_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MIN__;
            }
        }
    }
    g.sync();
    max = findMaxValue(g, buffer);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MIN_HD_IDX] = max;
    }

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MAX_GC_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MAX__;
            }
        }
    }
    g.sync();
    min = findMinValue(g, buffer);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MAX_GC_IDX] = min;
    }

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MAX_SL_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MAX__;
            }
        }
    }
    g.sync();
    min = findMinValue(g, buffer);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MAX_SL_IDX] = min;
    }

    return;
}

/* TODO :
Generation 마다 rank 0 에서 ideal point 업데이트 하기
    이거 non-dominated sorting 에서 업데이트 된 것 가지고 판독 필요
    global memory 에 버퍼 잡아논 것 있으니가 0번부터 sorting index 에서 읽어서 버퍼에 저장하고, 나머지는 안하고 해서 처리하면 가능.
*/

#endif
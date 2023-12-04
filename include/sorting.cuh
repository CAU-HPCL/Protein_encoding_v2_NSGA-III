#ifndef SORTING_H
#define SORTING_H

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <python3.10/Python.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "../include/common.cuh"
#include "../include/utils.cuh"

using namespace cooperative_groups;

#define _CRT_SECURE_NO_WARNINGS


__host__ void getReferencePointsEnergy(float *const h_reference_points, const int obj_num, const int ref_num)
{
    FILE *fp;
    char buffer[128];
    char py_command[128] = "python3 ./reference_points_energy.py";
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

__host__ void getReferencePointsDasDennis(float *const h_reference_points, const int obj_num, const int partition_num)
{
    FILE *fp;
    char buffer[128];
    char py_command[128] = "python3 ./reference_points_dasdennis.py";
    char tmp_str1[64];
    char tmp_str2[64];
    int i, j;

    sprintf(tmp_str1, " %d", obj_num);
    sprintf(tmp_str2, " %d", partition_num);
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
    if ((new_obj_val[MIN_CAI_IDX] == old_obj_val[MIN_CAI_IDX]) &&
        (new_obj_val[MIN_HD_IDX] == old_obj_val[MIN_HD_IDX]) &&
        (new_obj_val[MIN_CBP_IDX] == old_obj_val[MIN_CBP_IDX]) &&
        (new_obj_val[MIN_HSC_IDX] == old_obj_val[MIN_HSC_IDX]) &&
        (new_obj_val[MAX_GC_IDX] == old_obj_val[MAX_GC_IDX]) &&
        (new_obj_val[MAX_SL_IDX] == old_obj_val[MAX_SL_IDX]))
        return false;
    else if ((new_obj_val[MIN_CAI_IDX] <= old_obj_val[MIN_CAI_IDX]) &&
             (new_obj_val[MIN_HD_IDX] <= old_obj_val[MIN_HD_IDX]) &&
             (new_obj_val[MIN_CBP_IDX] <= old_obj_val[MIN_CBP_IDX]) &&
             (new_obj_val[MIN_HSC_IDX] <= old_obj_val[MIN_HSC_IDX]) &&
             (new_obj_val[MAX_GC_IDX] <= old_obj_val[MAX_GC_IDX]) &&
             (new_obj_val[MAX_SL_IDX] <= old_obj_val[MAX_SL_IDX]))
        return true;
    else
        return false;
}

__device__ void updateIdealValue(grid_group g, const float *obj_val, float *buffer, const int *d_sorted_array, const int *d_rank_count, int *index_num)
{
    int cycle_partition_num = ((c_N * 2 + OBJECTIVE_NUM) % g.size() == 0) ? ((c_N * 2 + OBJECTIVE_NUM) / g.size()) : ((c_N * 2 + OBJECTIVE_NUM) / g.size()) + 1;
    int g_tid;
    int i;
    float min;

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MIN_CAI_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MAX__;
            }
        }
    }
    g.sync();
    min = findMinValue(g, buffer, index_num);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MIN_CAI_IDX] = min;
    }
    g.sync();

    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MIN_CBP_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MAX__;
            }
        }
    }
    g.sync();
    min = findMinValue(g, buffer, index_num);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MIN_CBP_IDX] = min;
    }
    g.sync();


    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MIN_HSC_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MAX__;
            }
        }
    }
    g.sync();
    min = findMinValue(g, buffer, index_num);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MIN_HSC_IDX] = min;
    }
    g.sync();


    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
        {
            if (g_tid < d_rank_count[0])
            {
                buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MIN_HD_IDX];
            }
            else
            {
                buffer[g_tid] = __FLT_MAX__;
            }
        }
    }
    g.sync();
    min = findMinValue(g, buffer, index_num);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MIN_HD_IDX] = min;
    }
    g.sync();


    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
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
    min = findMinValue(g, buffer, index_num);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MAX_GC_IDX] = min;
    }
    g.sync();


    for (i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
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
    min = findMinValue(g, buffer, index_num);
    if (g.thread_rank() == 0)
    {
        estimated_ideal_value[MAX_SL_IDX] = min;
    }
    g.sync();


    return;
}

__device__ float ASF(const float *obj_val)
{
    int i;
    float tmp;
    float max = __FLT_MIN__;

    for (i = 0; i < OBJECTIVE_NUM; i++)
    {
        tmp = (obj_val[i] - estimated_ideal_value[i]) / weight_vector[i];
        if (tmp > max)
        {
            max = tmp;
        }
    }

    return max;
}

__device__ void replace_zero_gpu(grid_group g, int rows, int columns, int column)
{
    if (fabs(AB[column * columns + column]) <= 1e-6)
    {
        int row = column;
        for (; row < rows; row++)
        {
            if (fabs(AB[row * columns + column]) > 1e-6)
                break;
        }

        if (g.thread_rank() + column < columns)
        {
            int zero = column * columns + column + g.thread_rank();
            int chosen = row * columns + column + g.thread_rank();
            AB[zero] += AB[chosen];
        }
    }

    return;
}

__device__ void column_elimination_gpu(grid_group g, int rows, int columns, int column)
{
    if (g.thread_rank() < (rows - 1 - column) * (columns - column))
    {
        int el_row = column + g.thread_rank() / (columns - column) + 1;
        int el_col = column + g.thread_rank() % (columns - column);
        int el = el_col + el_row * columns;
        int upper_el = el_col + column * columns;

        int main_el = column + column * columns;
        int main2_el = column + el_row * columns;
        float f = AB[main2_el] / AB[main_el];

        AB[el] -= f * AB[upper_el];
    }

    return;
}

__device__ void reverse_row_elimination(grid_group g, int rows, int columns, int row)
{
    int cols = columns - 2 - row;
    int start_index = row * columns + row + 1;
    int j = cols % 2;
    bool check = true;

    for (int i = cols / 2; i > 0; i /= 2)
    {
        if (g.thread_rank() >= i)
        {
            check = false;
        }

        if (check)
        {
            AB[start_index + g.thread_rank()] += (AB[start_index + g.thread_rank() + i + j]);
            AB[start_index + g.thread_rank() + i + j] = 0.f;

            if (j == 1)
            {
                i++;
            }
            j = i % 2;
        }
        else
        {
            if (j == 1)
            {
                i++;
            }
            j = i % 2;
        }
        g.sync();
    }

    if (check)
    {
        int x_el = (row + 1) * columns - 1;
        int diag_el = row * columns + row;

        if (diag_el + 1 != x_el)
        {
            AB[x_el] -= AB[diag_el + 1];
            AB[diag_el + 1] = 0.f;
        }

        AB[x_el] /= AB[diag_el];
        AB[diag_el] = 1.0f;
    }

    return;
}

__device__ void multiple_column(grid_group g, int rows, int columns, int row)
{
    if (g.thread_rank() < row)
    {
        AB[(g.thread_rank() * columns) + row] *= AB[columns * (row + 1) - 1];
    }
    return;
}

__device__ void GaussianElimination(grid_group g)
{
    const int rows = OBJECTIVE_NUM;
    const int cols = OBJECTIVE_NUM + 1;

    int g_tid;
    int cycle_partition_num = ((rows * cols) % g.size() == 0) ? (rows * cols) / g.size() : (rows * cols) / g.size() + 1;

    for (int i = 0; i < cycle_partition_num; i++)
    {
        g_tid = g.size() * i + g.thread_rank();
        if (g_tid < (rows * cols))
        {
            if (((g_tid + 1) % cols) == 0)
            {
                AB[g_tid] = 1.0f;
            }
            else
            {
                AB[g_tid] = extreme_points[g_tid / cols][g_tid % cols];
            }
        }
    }
    g.sync();

    for (int column = 0; column < cols - 1; column++)
    {
        replace_zero_gpu(g, rows, cols, column);
        g.sync();

        column_elimination_gpu(g, rows, cols, column);
        g.sync();
    }

    for (int row = rows - 1; row >= 0; row--)
    {
        reverse_row_elimination(g, rows, cols, row);
        g.sync();

        multiple_column(g, rows, cols, row);
        g.sync();
    }

    return;
}

__device__ void findExtremePoints(grid_group g, const float *obj_val, float *buffer, int *index_num, const int *d_sorted_array, const int *d_rank_count)
{
    int cycle_partition_num;
    int g_tid;
    int i, j;
    float asf_result;

    for (i = 0; i < OBJECTIVE_NUM; i++)
    {
        if (g.thread_rank() == 0)
        {
            for (j = 0; j < OBJECTIVE_NUM; j++)
            {
                if (i == j)
                {
                    weight_vector[j] = 1.f;
                }
                else
                {
                    weight_vector[j] = f_precision;
                }
            }
        }
        g.sync();

        cycle_partition_num = ((c_N * 2 + OBJECTIVE_NUM) % g.size() == 0) ? ((c_N * 2 + OBJECTIVE_NUM) / g.size()) : ((c_N * 2 + OBJECTIVE_NUM) / g.size()) + 1;
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
            {
                buffer[g_tid] = __FLT_MAX__;
                index_num[g_tid] = EMPTY;
            }
        }
        g.sync();

        cycle_partition_num = ((d_rank_count[0] + OBJECTIVE_NUM) % g.size() == 0) ? ((d_rank_count[0] + OBJECTIVE_NUM) / g.size()) : ((d_rank_count[0] + OBJECTIVE_NUM) / g.size()) + 1;
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < (d_rank_count[0] + OBJECTIVE_NUM))
            {
                if (g_tid < d_rank_count[0])
                {
                    asf_result = ASF(&obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid]]);
                    index_num[g_tid] = d_sorted_array[g_tid];
                }
                else
                {
                    asf_result = ASF(extreme_points[g_tid - rank_count]);
                    index_num[g_tid] = (c_N * 2) + (g_tid - rank_count);
                }
                buffer[g_tid] = asf_result;
            }
        }
        g.sync();

        findMinValue(g, buffer, index_num);
        g.sync();

        if (g.thread_rank() == 0)
        {
            if (index_num[0] >= (c_N * 2))
            {
                for (j = 0; j < OBJECTIVE_NUM; j++)
                {
                    extreme_points[i][j] = extreme_points[index_num[0] - (c_N * 2)][j];
                }
            }
            else
            {
                for (j = 0; j < OBJECTIVE_NUM; j++)
                {
                    extreme_points[i][j] = obj_val[OBJECTIVE_NUM * index_num[0] + j];
                }
            }
        }
        g.sync();
    }

    return;
}

__device__ void updateNadirValue_MNDF(grid_group g, const float *obj_val, float *buffer, const int *d_sorted_array, const int *d_rank_count, int *index_num)
{
    int cycle_partition_num = ((c_N * 2 + OBJECTIVE_NUM) % g.size() == 0) ? ((c_N * 2 + OBJECTIVE_NUM) / g.size()) : ((c_N * 2 + OBJECTIVE_NUM) / g.size()) + 1;
    int g_tid;
    int i, j;
    float max;

    i = 0;
    while (true)
    {
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
            {
                if (g_tid < d_rank_count[i])
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

        max = findMaxValue(g, buffer, index_num);
        g.sync();
        if ((max - estimated_ideal_value[MIN_CAI_IDX]) >= f_precision)
        {
            break;
        }
        i++;
    }
    if (g.thread_rank() == 0)
    {
        estimated_nadir_value[MIN_CAI_IDX] = max;
    }
    g.sync();


    i = 0;
    while (true)
    {
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
            {
                if (g_tid < d_rank_count[i])
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

        max = findMaxValue(g, buffer, index_num);
        g.sync();
        if ((max - estimated_ideal_value[MIN_CBP_IDX]) >= f_precision)
        {
            break;
        }
        i++;
    }
    if (g.thread_rank() == 0)
    {
        estimated_nadir_value[MIN_CBP_IDX] = max;
    }
    g.sync();


    i = 0;
    while (true)
    {
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
            {
                if (g_tid < d_rank_count[i])
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

        max = findMaxValue(g, buffer, index_num);
        g.sync();
        if ((max - estimated_ideal_value[MIN_HSC_IDX]) >= f_precision)
        {
            break;
        }
        i++;
    }
    if (g.thread_rank() == 0)
    {
        estimated_nadir_value[MIN_HSC_IDX] = max;
    }
    g.sync();


    i = 0;
    while (true)
    {
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
            {
                if (g_tid < d_rank_count[i])
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

        max = findMaxValue(g, buffer, index_num);
        g.sync();
        if ((max - estimated_ideal_value[MIN_HD_IDX]) >= f_precision)
        {
            break;
        }
        i++;
    }
    if (g.thread_rank() == 0)
    {
        estimated_nadir_value[MIN_HD_IDX] = max;
    }
    g.sync();


    i = 0;
    while (true)
    {
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
            {
                if (g_tid < d_rank_count[i])
                {
                    buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MAX_GC_IDX];
                }
                else
                {
                    buffer[g_tid] = __FLT_MIN__;
                }
            }
        }
        g.sync();

        max = findMaxValue(g, buffer, index_num);
        g.sync();
        if ((max - estimated_ideal_value[MAX_GC_IDX]) >= f_precision)
        {
            break;
        }
        i++;
    }
    if (g.thread_rank() == 0)
    {
        estimated_nadir_value[MAX_GC_IDX] = max;
    }
    g.sync();


    i = 0;
    while (true)
    {
        for (j = 0; j < cycle_partition_num; j++)
        {
            g_tid = g.size() * j + g.thread_rank();
            if (g_tid < (c_N * 2 + OBJECTIVE_NUM))
            {
                if (g_tid < d_rank_count[i])
                {
                    buffer[g_tid] = obj_val[OBJECTIVE_NUM * d_sorted_array[g_tid] + MAX_SL_IDX];
                }
                else
                {
                    buffer[g_tid] = __FLT_MIN__;
                }
            }
        }
        g.sync();

        max = findMaxValue(g, buffer, index_num);
        g.sync();
        if ((max - estimated_ideal_value[MAX_SL_IDX]) >= f_precision)
        {
            break;
        }
        i++;
    }
    if (g.thread_rank() == 0)
    {
        estimated_nadir_value[MAX_SL_IDX] = max;
    }
    g.sync();


    return;
}

__device__ void updateNadirValue_ME(grid_group g, const float *obj_val, float *buffer, int *index_num, const int *d_sorted_array, const int *d_rank_count)
{
    int i;

    findExtremePoints(g, obj_val, buffer, index_num, d_sorted_array, d_rank_count);
    g.sync();

    if (g.thread_rank() < OBJECTIVE_NUM)
    {
        estimated_nadir_value[g.thread_rank()] = extreme_points[0][g.thread_rank()];
        for (i = 1; i < OBJECTIVE_NUM; i++)
        {
            if (estimated_nadir_value[g.thread_rank()] < extreme_points[i][g.thread_rank()])
            {
                estimated_nadir_value[g.thread_rank()] = extreme_points[i][g.thread_rank()];
            }
        }
    }
    g.sync();

    return;
}

__device__ void updateNadirValue_HYP(grid_group g, thread_group tb, const float *obj_val, float *buffer, const int *d_sorted_array, const int *d_rank_count, int *index_num)
{
    int i;
    float intercept;

    if (g.thread_rank() == 0)
    {
        HYP_EXCEPTION = false;
        g_mutex = 0;
    }
    g.sync();

    findExtremePoints(g, obj_val, buffer, index_num, d_sorted_array, d_rank_count);
    g.sync();

    GaussianElimination(g);
    g.sync();

    if (g.thread_rank() < OBJECTIVE_NUM)
    {
        float result = 0.f;
        for (i = 0; i < OBJECTIVE_NUM; i++)
        {
            result += AB[(OBJECTIVE_NUM + 1) * i + OBJECTIVE_NUM] * (extreme_points[g.thread_rank()][i] - estimated_ideal_value[i]);
        }

        if (fabs(result - 1.f) > f_precision || isnan(result) || isinf(result))
        {
            while (atomicCAS(&g_mutex, 0, 1) != 0)
            {
            }
            HYP_EXCEPTION = true;
            atomicExch(&g_mutex, 0);
        }

        intercept = 1.f / AB[(OBJECTIVE_NUM + 1) * g.thread_rank() + OBJECTIVE_NUM];
        if (intercept <= f_precision || isnan(intercept) || isinf(intercept))
        {
            while (atomicCAS(&g_mutex, 0, 1) != 0)
            {
            }
            HYP_EXCEPTION = true;
            atomicExch(&g_mutex, 0);
        }
    }
    g.sync();

    if (HYP_EXCEPTION)
    {
        updateNadirValue_MNDF(g, obj_val, buffer, d_sorted_array, d_rank_count, index_num);
    }
    else
    {
        if (g.thread_rank() < OBJECTIVE_NUM)
        {
            estimated_nadir_value[g.thread_rank()] = estimated_ideal_value[g.thread_rank()] + intercept;
        }
    }
    g.sync();

    return;
}

__device__ void nonDominatedSorting(grid_group g, const float *d_obj_val, int *d_sorted_array, bool *F_set, bool *Sp_set, int *d_np, int *d_rank_count)
{
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
                idx = atomicAdd(&rank_count, 1);
                d_sorted_array[idx] = g_tid;
                atomicAdd(&d_rank_count[cur_front], 1);
            }
        }
    }
    g.sync();

    if (rank_count == c_N)
    {
        if (g.thread_rank() == 0)
        {
            N_cut_check = true;
        }
        g.sync();

        return;
    }

    /* -------------------- After 1st front setting -------------------- */
    if (rank_count < c_N)
    {
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

    return;
}

__device__ void referenceBasedSorting(curandStateXORWOW *random_generator, grid_group g, thread_block tb, const float *d_obj_val, int *d_sorted_array, const int *d_rank_count, const float *d_reference_points, int *d_included_solution_num, int *d_not_included_solution_num, int *d_solution_index_for_sorting, float *d_dist_of_solution, float *s_buffer, int *s_index_num, float *s_normalized_obj_val)
{
    int i, j, k;
    int block_cycle_partition_num;
    int thread_cycle_partition_num;
    int block_id;
    int thread_id;

    if (g.thread_rank() == 0)
    {
        number_of_count = rank_count - d_rank_count[cur_front];
    }
    g.sync();

    block_cycle_partition_num = (rank_count % g.num_blocks() == 0) ? (rank_count / g.num_blocks()) : (rank_count / g.num_blocks()) + 1;
    thread_cycle_partition_num = (c_ref_points_num % tb.size() == 0) ? (c_ref_points_num / tb.size()) : (c_ref_points_num / tb.size()) + 1;
    for (i = 0; i < block_cycle_partition_num; i++)
    {
        block_id = g.num_blocks() * i + g.block_rank();
        if (block_id < rank_count)
        {
            if (tb.thread_rank() < OBJECTIVE_NUM)
            {
                s_normalized_obj_val[tb.thread_rank()] = (d_obj_val[d_sorted_array[block_id] * OBJECTIVE_NUM + tb.thread_rank()] - estimated_ideal_value[tb.thread_rank()]) / (estimated_nadir_value[tb.thread_rank()] - estimated_ideal_value[tb.thread_rank()]);
            }
            tb.sync();

            s_buffer[tb.thread_rank()] = __FLT_MAX__;
            for (j = 0; j < thread_cycle_partition_num; j++)
            {
                float tmp_result;
                thread_id = tb.size() * j + tb.thread_rank();
                if (thread_id < c_ref_points_num)
                {
                    tmp_result = perpendicularDistance(s_normalized_obj_val, &d_reference_points[OBJECTIVE_NUM * thread_id]);
                    if (tmp_result < s_buffer[tb.thread_rank()])
                    {
                        s_buffer[tb.thread_rank()] = tmp_result;
                        s_index_num[tb.thread_rank()] = thread_id;
                    }
                }
            }
            tb.sync();

            j = tb.size() / 2;
            while (true)
            {
                if ((tb.thread_rank() < j) && (s_buffer[tb.thread_rank() + j] < s_buffer[tb.thread_rank()]))
                {
                    s_buffer[tb.thread_rank()] = s_buffer[tb.thread_rank() + j];
                    s_index_num[tb.thread_rank()] = s_index_num[tb.thread_rank() + j];
                }
                tb.sync();

                if (j == 1)
                {
                    break;
                }

                if ((j % 2 == 1) && (tb.thread_rank() == 0))
                {
                    if (s_buffer[j - 1] < s_buffer[0])
                    {
                        s_buffer[0] = s_buffer[j - 1];
                        s_index_num[0] = s_index_num[j - 1];
                    }
                }
                tb.sync();

                j /= 2;
            }
            tb.sync();

            if (tb.thread_rank() == 0)
            {
                d_dist_of_solution[d_sorted_array[block_id]] = s_buffer[0];
                if (block_id < (rank_count - d_rank_count[cur_front]))
                {
                    atomicAdd(&d_included_solution_num[s_index_num[0]], 1);
                }
                else
                {
                    int tmp_index = atomicAdd(&d_not_included_solution_num[s_index_num[0]], 1);
                    d_solution_index_for_sorting[(c_N * 2) * s_index_num[0] + tmp_index] = d_sorted_array[block_id];
                }
            }
            tb.sync();
        }
    }
    g.sync();

    block_cycle_partition_num = (c_ref_points_num % g.num_blocks() == 0) ? (c_ref_points_num / g.num_blocks()) : (c_ref_points_num / g.num_blocks()) + 1;
    thread_cycle_partition_num = ((c_N * 2) % tb.size() == 0) ? ((c_N * 2) / tb.size()) : ((c_N * 2) / tb.size()) + 1;
    i = 0;
    while (true)
    {
        for (j = 0; j < block_cycle_partition_num; j++)
        {
            block_id = g.num_blocks() * j + g.block_rank();
            if ((block_id < c_ref_points_num) && (d_included_solution_num[block_id] == i) && (d_not_included_solution_num[block_id] > 0))
            {
                if (i == 0)
                {
                    s_buffer[tb.thread_rank()] = __FLT_MAX__;
                    int tmp_sol_indicate = EMPTY;
                    for (k = 0; k < thread_cycle_partition_num; k++)
                    {
                        thread_id = tb.size() * k + tb.thread_rank();
                        if ((thread_id < (c_N * 2)) && (d_solution_index_for_sorting[(c_N * 2) * block_id + thread_id] != EMPTY))
                        {
                            if (d_dist_of_solution[d_solution_index_for_sorting[(c_N * 2) * block_id + thread_id]] < s_buffer[tb.thread_rank()])
                            {
                                s_buffer[tb.thread_rank()] = d_dist_of_solution[d_solution_index_for_sorting[(c_N * 2) * block_id + thread_id]];
                                s_index_num[tb.thread_rank()] = d_solution_index_for_sorting[(c_N * 2) * block_id + thread_id];
                                tmp_sol_indicate = thread_id;
                            }
                        }
                    }
                    tb.sync();

                    k = tb.size() / 2;
                    while (true)
                    {
                        if ((tb.thread_rank() < k) && (s_buffer[tb.thread_rank() + k] < s_buffer[tb.thread_rank()]))
                        {
                            s_buffer[tb.thread_rank()] = s_buffer[tb.thread_rank() + k];
                            s_index_num[tb.thread_rank()] = s_index_num[tb.thread_rank() + k];
                        }
                        tb.sync();

                        if (k == 1)
                        {
                            break;
                        }

                        if ((k % 2 == 1) && (tb.thread_rank() == 0))
                        {
                            if (s_buffer[k - 1] < s_buffer[0])
                            {
                                s_buffer[0] = s_buffer[k - 1];
                                s_index_num[0] = s_index_num[k - 1];
                            }
                        }
                        tb.sync();

                        k /= 2;
                    }
                    tb.sync();

                    if ((tmp_sol_indicate != EMPTY) && (s_index_num[0] == d_solution_index_for_sorting[(c_N * 2) * block_id + tmp_sol_indicate]))
                    {
                        int tmp_index = atomicAdd(&number_of_count, 1);
                        d_sorted_array[tmp_index] = s_index_num[0];
                        d_included_solution_num[block_id] += 1;
                        d_not_included_solution_num[block_id] -= 1;
                        d_solution_index_for_sorting[(c_N * 2) * block_id + tmp_sol_indicate] = d_solution_index_for_sorting[(c_N * 2) * block_id + d_not_included_solution_num[block_id]];
                        d_solution_index_for_sorting[(c_N * 2) * block_id + d_not_included_solution_num[block_id]] = EMPTY;
                    }
                    tb.sync();
                }
                else
                {
                    if (tb.thread_rank() == 0)
                    {
                        int tmp_index = atomicAdd(&number_of_count, 1);
                        int tmp_sol_indicate;
                        do
                        {
                            tmp_sol_indicate = (int)(curand_uniform(random_generator) * d_not_included_solution_num[block_id]);
                        } while (tmp_sol_indicate == d_not_included_solution_num[block_id]);
                        d_sorted_array[tmp_index] = d_solution_index_for_sorting[(c_N * 2) * block_id + tmp_sol_indicate];
                        d_included_solution_num[block_id] += 1;
                        d_not_included_solution_num[block_id] -= 1;
                        d_solution_index_for_sorting[(c_N * 2) * block_id + tmp_sol_indicate] = d_solution_index_for_sorting[(c_N * 2) * block_id + d_not_included_solution_num[block_id]];
                        d_solution_index_for_sorting[(c_N * 2) * block_id + d_not_included_solution_num[block_id]] = EMPTY;
                    }
                    tb.sync();
                }
            }
        }
        g.sync();

        if (number_of_count >= c_N)
        {
            break;
        }

        i++;
    }

    return;
}

#endif
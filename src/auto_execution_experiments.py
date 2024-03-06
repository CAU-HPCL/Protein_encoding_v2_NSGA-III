import subprocess
import os
import math
import sys
import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

REPEAT_NUM = 10

MY_PROGRAM = "/home/jeus5771/Dynamic_Check/Protein_NSGA3/src/Protein_NSGA3"
PROTEIN_FILE_PATH = [
    # "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/Q5VZP5.fasta.txt", 
    # "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/A4Y1B6.fasta.txt",
    # "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/B3LS90.fasta.txt",
    # "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/B4TWR7.fasta.txt",
    # "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/Q91X51.fasta.txt",
    # "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/Q89BP2.fasta.txt",
    # "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/A6L9J9.fasta.txt",
    # "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/Q88X33.fasta.txt",
    "/home/jeus5771/Dynamic_Check/Protein_NSGA3/Protein_FASTA/B7KHU9.fasta.txt",
    ]
PROTEIN = [
    # "Q5VZP5",
    # "A4Y1B6",
    # "B3LS90",
    # "B4TWR7",
    # "Q91X51",
    # "Q89BP2",
    # "A6L9J9",
    # "Q88X33",
    "B7KHU9"
]

POPULATION_SIZE = ["100", "200", "400"]
CYCLE_NUM = ["100", "200", "400", "800", "1600", "3200", "6400", "12800"]
# CDS_NUM = ["2", "3", "4", "5", "6", "7", "8", "9", "10"]
CDS_NUM = ["10"]
MUTATION_PROB = ["dynamic"]

FOLDER_PATH = "/home/jeus5771/Dynamic_Check/Protein_NSGA3/src/Check/"

# MOVNS_MOSFLA_IDEAL = (1, 0.4, 0) # mCAI, mHD, MLRCS
# MOVNS_MOSFLA_NADIR = (0, 0, 1)
# MaOMPE_IDEAL = (1, 0.4, 0, 0) # mCAI, mHD, MGC, MSL
# MaOMPE_NADIR = (0, 0, 0.6, 1)
# MOBOA_IDEAL = (1, 0.5, 0) # mCAI, mHD, MGC3
# MOBOA_NADIR = (0, 0, 0.6)
My_IDEAL = (1, 0.2, 0.2, 0.5, 0, 0) # mCAI, mCPB, mHSC, mHD, MGC3, MSL
My_NADIR = (0, -0.15, 0, 0, 0.6, 1)


def cal_3_volume(val_list):
    with open('tmp.txt', 'w') as file:
        for sublist in val_list:
            file.write(' '.join(map(str, sublist)) + '\n')
    command = ["./hv", "-r", "1 1 1", "tmp.txt"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = float(result.stdout)
    return result

def cal_4_volume(val_list):
    with open('tmp.txt', 'w') as file:
        for sublist in val_list:
            file.write(' '.join(map(str, sublist)) + '\n')
    command = ["./hv", "-r", "1 1 1 1", "tmp.txt"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = float(result.stdout)
    return result

def cal_6_volume(val_list):
    with open('tmp.txt', 'w') as file:
        for sublist in val_list:
            file.write(' '.join(map(str, sublist)) + '\n')
    command = ["./hv", "-r", "1 1 1 1 1 1", "tmp.txt"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = float(result.stdout)
    return result

def cal_3_spread(val_list):
    f_max = [-sys.float_info.max, -sys.float_info.max, -sys.float_info.max]
    f_min = [sys.float_info.max, sys.float_info.max, sys.float_info.max]
    for i in range(len(val_list)):
        for j in range(3):
            if f_max[j] < val_list[i][j]:
                f_max[j] = val_list[i][j]
            if f_min[j] > val_list[i][j]:
                f_min[j] = val_list[i][j]
    
    result = 0
    for i in range(3):
        result += math.pow(f_max[i] - f_min[i], 2)
    result = math.sqrt(result)

    return result

def cal_4_spread(val_list):
    f_max = [-sys.float_info.max, -sys.float_info.max, -sys.float_info.max, -sys.float_info.max]
    f_min = [sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max]
    for i in range(len(val_list)):
        for j in range(4):
            if f_max[j] < val_list[i][j]:
                f_max[j] = val_list[i][j]
            if f_min[j] > val_list[i][j]:
                f_min[j] = val_list[i][j]
    
    result = 0
    for i in range(4):
        result += math.pow(f_max[i] - f_min[i], 2)
    result = math.sqrt(result)

    return result

def cal_6_spread(val_list):
    f_max = [-sys.float_info.max, -sys.float_info.max, -sys.float_info.max, -sys.float_info.max, -sys.float_info.max, -sys.float_info.max]
    f_min = [sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max]
    for i in range(len(val_list)):
        for j in range(6):
            if f_max[j] < val_list[i][j]:
                f_max[j] = val_list[i][j]
            if f_min[j] > val_list[i][j]:
                f_min[j] = val_list[i][j]
    
    result = 0
    for i in range(6):
        result += math.pow(f_max[i] - f_min[i], 2)
    result = math.sqrt(result)

    return result

def cal_3_dist(val_list):
    result = sys.float_info.max
    for i in range(len(val_list)):
        tmp = math.sqrt(math.pow(val_list[i][0], 2) + math.pow(val_list[i][1], 2) + math.pow(val_list[i][2], 2))
        if tmp < result:
            result = tmp
    return result

def cal_4_dist(val_list):
    result = sys.float_info.max
    for i in range(len(val_list)):
        tmp = math.sqrt(math.pow(val_list[i][0], 2) + math.pow(val_list[i][1], 2) + math.pow(val_list[i][2], 2) + math.pow(val_list[i][3], 2))
        if tmp < result:
            result = tmp
    return result

def cal_6_dist(val_list):
    result = sys.float_info.max
    for i in range(len(val_list)):
        tmp = math.sqrt(math.pow(val_list[i][0], 2) + math.pow(val_list[i][1], 2) + math.pow(val_list[i][2], 2) + math.pow(val_list[i][3], 2) + math.pow(val_list[i][4], 2) + math.pow(val_list[i][5], 2))
        if tmp < result:
            result = tmp
    return result

# --------------------------------------------------------------------------------------------------------------------

for i in range(len(PROTEIN_FILE_PATH)):
    if not os.path.exists(FOLDER_PATH + PROTEIN[i]):
        os.makedirs(FOLDER_PATH + PROTEIN[i])
    if not os.path.exists(FOLDER_PATH + PROTEIN[i] + '/' + 'ObjVal'):
        os.makedirs(FOLDER_PATH + PROTEIN[i] + '/' + 'ObjVal')
    if not os.path.exists(FOLDER_PATH + PROTEIN[i] + '/' + 'CompResult'):
        os.makedirs(FOLDER_PATH + PROTEIN[i] + '/' + 'CompResult')
    for pop_size in POPULATION_SIZE:
        for cycles in CYCLE_NUM:
            for muation_prob in MUTATION_PROB:
                with open(FOLDER_PATH + PROTEIN[i] + '/CompResult/' + str(pop_size) + '_' + str(cycles) + '_' + str(muation_prob) + '.txt', 'w') as comp_file:
                    average_time = 0
                    # average_movns_mosfla_hypervolume = 0
                    # average_movns_mosfla_maximum_spread = 0
                    # average_movns_mosfla_minimum_distance = 0
                    # average_maompe_hypervolume = 0
                    # average_maompe_maximum_spread = 0
                    # average_maompe_minimum_distance = 0
                    # average_moboa_hypervolume = 0
                    # average_moboa_maximum_spread = 0
                    # average_moboa_minimum_distance = 0
                    average_my_hypervolume = 0
                    average_my_maximum_spread = 0
                    average_my_minimum_distance = 0
                    average_my_best_obj_val = [0, 0, 0, 0, 0, 0]
                    average_my_worst_obj_val = [0, 0, 0, 0, 0, 0]
                    for r in range(REPEAT_NUM):
                        with open(FOLDER_PATH + PROTEIN[i] + '/ObjVal/' +  str(pop_size) + '_' + str(cycles) + '_' + str(muation_prob) + '_' + str(r + 1) + '.txt', 'w') as obj_file:
                            this_time = 0
                            # this_movns_mosfla_hypervolume = 0
                            # this_movns_mosfla_maximum_spread = 0
                            # this_movns_mosfla_minimum_distance = 0
                            # this_maompe_hypervolume = 0
                            # this_maompe_maximum_spread = 0
                            # this_maompe_minimum_distance = 0
                            # this_moboa_hypervolume = 0
                            # this_moboa_maximum_spread = 0
                            # this_moboa_minimum_distance = 0
                            this_my_hypervolume = 0
                            this_my_maximum_spread = 0
                            this_my_minimum_distance = 0
                            this_my_best_obj_val = [-sys.float_info.max, -sys.float_info.max, -sys.float_info.max, -sys.float_info.max, sys.float_info.max, sys.float_info.max]
                            this_my_worst_obj_val = [sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max, -sys.float_info.max, -sys.float_info.max]
                            
                            command = [MY_PROGRAM] + [PROTEIN_FILE_PATH[i]] + [pop_size] + [cycles] + [CDS_NUM[i]] + [muation_prob]
                            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            if result.returncode == 0:
                                print(str(command) + " Complete")
                            else:
                                print(str(command) + " Fail")
                            
                            check = True

                            obj_val = list()    # [mCAI, mCPB, mHSC, mHD, MGC3, MSL, MGC, MLRCS]
                            # norm_movns_mosfla = list()
                            # norm_maompe = list()
                            # norm_moboa = list()
                            norm_my = list()

                            for line in result.stdout.splitlines():
                                if(line == 'end'):
                                    check = False
                                    continue
                                if check:
                                    this_time = float(line)
                                else:
                                    obj_file.write(line + '\n')
                                    obj_val += [[float(x) for x in line.split()]]


                            for val in obj_val:
                                # norm_movns_mosfla.append([(val[0] - MOVNS_MOSFLA_IDEAL[0]) / (MOVNS_MOSFLA_NADIR[0] - MOVNS_MOSFLA_IDEAL[0]), 
                                #                       (val[3] - MOVNS_MOSFLA_IDEAL[1]) / (MOVNS_MOSFLA_NADIR[1] - MOVNS_MOSFLA_IDEAL[1]), 
                                #                       (val[7] - MOVNS_MOSFLA_IDEAL[2]) / (MOVNS_MOSFLA_NADIR[2] - MOVNS_MOSFLA_IDEAL[2]), 
                                #                       ])
                                # norm_maompe.append([(val[0] - MaOMPE_IDEAL[0]) / (MaOMPE_NADIR[0] - MaOMPE_IDEAL[0]), 
                                #                 (val[3] - MaOMPE_IDEAL[1]) / (MaOMPE_NADIR[1] - MaOMPE_IDEAL[1]), 
                                #                 (val[6] - MaOMPE_IDEAL[2]) / (MaOMPE_NADIR[2] - MaOMPE_IDEAL[2]), 
                                #                 (val[5] - MaOMPE_IDEAL[3]) / (MaOMPE_NADIR[3] - MaOMPE_IDEAL[3])])
                                # norm_moboa.append([(val[0] - MOBOA_IDEAL[0])/ (MOBOA_NADIR[0] - MOBOA_IDEAL[0]), 
                                #                (val[3] - MOBOA_IDEAL[1])/ (MOBOA_NADIR[1] - MOBOA_IDEAL[1]), 
                                #                (val[4] - MOBOA_IDEAL[2])/ (MOBOA_NADIR[2] - MOBOA_IDEAL[2])])
                                norm_my.append([(val[0] - My_IDEAL[0]) / (My_NADIR[0] - My_IDEAL[0]), 
                                            (val[1] - My_IDEAL[1]) / (My_NADIR[1] - My_IDEAL[1]), 
                                            (val[2] - My_IDEAL[2]) / (My_NADIR[2] - My_IDEAL[2]), 
                                            (val[3] - My_IDEAL[3]) / (My_NADIR[3] - My_IDEAL[3]), 
                                            (val[4] - My_IDEAL[4]) / (My_NADIR[4] - My_IDEAL[4]), 
                                            (val[5] - My_IDEAL[5]) / (My_NADIR[5] - My_IDEAL[5])])
                                
                            # for_movns_mosfla_spread = list()
                            # for_maompe_spread = list()
                            # for_moboa_spread = list()
                            for_my_spread = list()

                            # movns_mosfla_solutions_list = [Individual() for _ in range(len(norm_movns_mosfla))]
                            # for j in range(len(norm_movns_mosfla)):
                            #     tmp_list = list()
                            #     movns_mosfla_solutions_list[j].X = j
                            #     tmp_list.append(-(1 - norm_movns_mosfla[j][0]))
                            #     tmp_list.append(-(1 - norm_movns_mosfla[j][1]))
                            #     tmp_list.append(norm_movns_mosfla[j][2])
                            #     movns_mosfla_solutions_list[j].F = np.array(tmp_list)
                            # movns_mosfla_pop = Population(individuals=movns_mosfla_solutions_list)
                            # movns_mosfla_F = movns_mosfla_pop.get("F")
                            # movns_mosfla_fronts = NonDominatedSorting().do(movns_mosfla_F)

                            # maompe_solutions_list = [Individual() for _ in range(len(norm_maompe))]
                            # for j in range(len(norm_maompe)):
                            #     tmp_list = list()
                            #     maompe_solutions_list[j].X = j
                            #     tmp_list.append(-(1 - norm_maompe[j][0]))
                            #     tmp_list.append(-(1 - norm_maompe[j][1]))
                            #     tmp_list.append(norm_maompe[j][2])
                            #     tmp_list.append(norm_maompe[j][3])
                            #     maompe_solutions_list[j].F = np.array(tmp_list)
                            # maompe_pop = Population(individuals=maompe_solutions_list)
                            # maompe_F = maompe_pop.get("F")
                            # maompe_fronts = NonDominatedSorting().do(maompe_F)

                            # moboa_solutions_list = [Individual() for _ in range(len(norm_moboa))]
                            # for j in range(len(norm_moboa)):
                            #     tmp_list = list()
                            #     moboa_solutions_list[j].X = j
                            #     tmp_list.append(-(1 - norm_moboa[j][0]))
                            #     tmp_list.append(-(1 - norm_moboa[j][1]))
                            #     tmp_list.append(norm_moboa[j][2])
                            #     moboa_solutions_list[j].F = np.array(tmp_list)
                            # moboa_pop = Population(individuals=moboa_solutions_list)
                            # moboa_F = moboa_pop.get("F")
                            # moboa_fronts = NonDominatedSorting().do(moboa_F)

                            my_solutions_list = [Individual() for _ in range(len(norm_my))]
                            for j in range(len(norm_my)):
                                tmp_list = list()
                                my_solutions_list[j].X = j
                                tmp_list.append(-(1 - norm_my[j][0]))
                                tmp_list.append(-(1 - norm_my[j][1]))
                                tmp_list.append(-(1 - norm_my[j][2]))
                                tmp_list.append(-(1 - norm_my[j][3]))
                                tmp_list.append(norm_my[j][4])
                                tmp_list.append(norm_my[j][5])
                                my_solutions_list[j].F = np.array(tmp_list)
                            my_pop = Population(individuals=my_solutions_list)
                            my_F = my_pop.get("F")
                            my_fronts = NonDominatedSorting().do(my_F)


                            # for j in range(len(norm_movns_mosfla)):
                            #     for k in range(3):
                            #         if norm_movns_mosfla[j][k] <= 0:
                            #             norm_movns_mosfla[j][k] = 0
                            #         if norm_movns_mosfla[j][k] >= 1:
                            #             norm_movns_mosfla[j][k] = 0.999999

                            # for j in range(len(norm_maompe)):
                            #     for k in range(4):
                            #         if norm_maompe[j][k] <= 0:
                            #             norm_maompe[j][k] = 0
                            #         if norm_maompe[j][k] >= 1:
                            #             norm_maompe[j][k] = 0.999999

                            # for j in range(len(norm_moboa)):
                            #     for k in range(3):
                            #         if norm_moboa[j][k] <= 0:
                            #             norm_moboa[j][k] = 0
                            #         if norm_moboa[j][k] >= 1:
                            #             norm_moboa[j][k] = 0.999999

                            for j in range(len(norm_my)):
                                for k in range(6):
                                    if norm_my[j][k] <= 0:
                                        norm_my[j][k] = 0
                                    if norm_my[j][k] >= 1:
                                        norm_my[j][k] = 0.999999                            

                            # for j in range(len(movns_mosfla_fronts[0])):
                            #     for_movns_mosfla_spread.append(norm_movns_mosfla[movns_mosfla_fronts[0][j]])
                            # for j in range(len(maompe_fronts[0])):
                            #     for_maompe_spread.append(norm_maompe[maompe_fronts[0][j]])
                            # for j in range(len(moboa_fronts[0])):
                            #     for_moboa_spread.append(norm_moboa[moboa_fronts[0][j]])
                            for j in range(len(my_fronts[0])):
                                for_my_spread.append(norm_my[my_fronts[0][j]])
                            
                            # this_movns_mosfla_hypervolume = cal_3_volume(norm_movns_mosfla)
                            # this_movns_mosfla_maximum_spread = cal_3_spread(for_movns_mosfla_spread)
                            # this_movns_mosfla_minimum_distance = cal_3_dist(norm_movns_mosfla)
                            # this_maompe_hypervolume = cal_4_volume(norm_maompe)
                            # this_maompe_maximum_spread = cal_4_spread(for_maompe_spread)
                            # this_maompe_minimum_distance = cal_4_dist(norm_maompe)
                            # this_moboa_hypervolume = cal_3_volume(norm_moboa)
                            # this_moboa_maximum_spread = cal_3_spread(for_moboa_spread)
                            # this_moboa_minimum_distance = cal_3_dist(norm_moboa)
                            this_my_hypervolume = cal_6_volume(norm_my)
                            this_my_maximum_spread = cal_6_spread(for_my_spread)
                            this_my_minimum_distance = cal_6_dist(norm_my)
                            
                            comp_file.write(f'Execution time : {this_time:.6f}\n')
                            # comp_file.write(f'Comp_MOVNS_MOSFLA : {this_movns_mosfla_hypervolume:.6f}\t{this_movns_mosfla_maximum_spread:.6f}\t{this_movns_mosfla_minimum_distance:.6f}\n')
                            # comp_file.write(f'Comp_MaOMPE : {this_maompe_hypervolume:.6f}\t{this_maompe_maximum_spread:.6f}\t{this_maompe_minimum_distance:.6f}\n')
                            # comp_file.write(f'Comp_MOBOA : {this_moboa_hypervolume:.6f}\t{this_moboa_maximum_spread:.6f}\t{this_moboa_minimum_distance:.6f}\n')
                            comp_file.write(f'My : {this_my_hypervolume:.6f}\t{this_my_maximum_spread:.6f}\t{this_my_minimum_distance:.6f}\n')

                            average_time += this_time
                            # average_movns_mosfla_hypervolume += this_movns_mosfla_hypervolume
                            # average_movns_mosfla_maximum_spread += this_movns_mosfla_maximum_spread
                            # average_movns_mosfla_minimum_distance += this_movns_mosfla_minimum_distance
                            # average_maompe_hypervolume += this_maompe_hypervolume
                            # average_maompe_maximum_spread += this_maompe_maximum_spread
                            # average_maompe_minimum_distance += this_maompe_minimum_distance
                            # average_moboa_hypervolume += this_moboa_hypervolume
                            # average_moboa_maximum_spread += this_moboa_maximum_spread
                            # average_moboa_minimum_distance += this_moboa_minimum_distance
                            average_my_hypervolume += this_my_hypervolume
                            average_my_maximum_spread += this_my_maximum_spread
                            average_my_minimum_distance += this_my_minimum_distance

                            for j in range(len(norm_my)):
                                for k in range(4):
                                    if this_my_best_obj_val[k] < (1 - norm_my[j][k]):
                                        this_my_best_obj_val[k] = (1 - norm_my[j][k])
                                    if this_my_worst_obj_val[k] > (1 - norm_my[j][k]):
                                        this_my_worst_obj_val[k] = (1 - norm_my[j][k])
                                for k in range(2):
                                    if this_my_best_obj_val[k + 4] > norm_my[j][k + 4]:
                                        this_my_best_obj_val[k + 4] = norm_my[j][k + 4]
                                    if this_my_worst_obj_val[k + 4] < norm_my[j][k + 4]:
                                        this_my_worst_obj_val[k + 4] = norm_my[j][k + 4]
                            
                            for j in range(6):
                                average_my_best_obj_val[j] += this_my_best_obj_val[j]
                                average_my_worst_obj_val[j] += this_my_worst_obj_val[j]
                            
                            comp_file.write(f'Best obj val : {this_my_best_obj_val[0]:.6f}\t{this_my_best_obj_val[1]:.6f}\t{this_my_best_obj_val[2]:.6f}\t{this_my_best_obj_val[3]:.6f}\t{this_my_best_obj_val[4]:.6f}\t{this_my_best_obj_val[5]:.6f}\n')
                            comp_file.write(f'Worst obj val : {this_my_worst_obj_val[0]:.6f}\t{this_my_worst_obj_val[1]:.6f}\t{this_my_worst_obj_val[2]:.6f}\t{this_my_worst_obj_val[3]:.6f}\t{this_my_worst_obj_val[4]:.6f}\t{this_my_worst_obj_val[5]:.6f}\n')
                            comp_file.write('\n')
                
                    comp_file.write(f'Execution time : {(average_time / REPEAT_NUM):.6f}\n')
                    # comp_file.write(f'Average Comp_MOVNS_MOSFLA : {(average_movns_mosfla_hypervolume / REPEAT_NUM):.6f}\t{(average_movns_mosfla_maximum_spread / REPEAT_NUM):.6f}\t{(average_movns_mosfla_minimum_distance / REPEAT_NUM):.6f}\n')
                    # comp_file.write(f'Average Comp_MaOMPE : {(average_maompe_hypervolume / REPEAT_NUM):.6f}\t{(average_maompe_maximum_spread / REPEAT_NUM):.6f}\t{(average_maompe_minimum_distance / REPEAT_NUM):.6f}\n')
                    # comp_file.write(f'Average Comp_MOBOA : {(average_moboa_hypervolume / REPEAT_NUM):.6f}\t{(average_moboa_maximum_spread / REPEAT_NUM):.6f}\t{(average_moboa_minimum_distance / REPEAT_NUM):.6f}\n')
                    comp_file.write(f'Average My : {(average_my_hypervolume / REPEAT_NUM):.6f}\t{(average_my_maximum_spread / REPEAT_NUM):.6f}\t{(average_my_minimum_distance / REPEAT_NUM):.6f}\n')
                    
                    for m in range(6):
                        average_my_best_obj_val[m] /= REPEAT_NUM
                        average_my_worst_obj_val[m] /= REPEAT_NUM

                    comp_file.write(f'Average Best obj val : {average_my_best_obj_val[0]:.6f}\t{average_my_best_obj_val[1]:.6f}\t{average_my_best_obj_val[2]:.6f}\t{average_my_best_obj_val[3]:.6f}\t{average_my_best_obj_val[4]:.6f}\t{average_my_best_obj_val[5]:.6f}\n')
                    comp_file.write(f'Average Worst obj val : {average_my_worst_obj_val[0]:.6f}\t{average_my_worst_obj_val[1]:.6f}\t{average_my_worst_obj_val[2]:.6f}\t{average_my_worst_obj_val[3]:.6f}\t{average_my_worst_obj_val[4]:.6f}\t{average_my_worst_obj_val[5]:.6f}\n')
                    comp_file.write('\n')

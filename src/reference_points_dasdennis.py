from pymoo.util.ref_dirs import get_reference_directions
import sys

if __name__ == "__main__":
    objective_num = (int)(sys.argv[1])
    partition_num = (int)(sys.argv[2])
    ref_dirs = get_reference_directions("das-dennis", objective_num, n_partitions = partition_num)
    num = 1
    for i in ref_dirs:
        for j in i:
            print(j)
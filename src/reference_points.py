from pymoo.util.ref_dirs import get_reference_directions
import sys

if __name__ == "__main__":
    objective_num = (int)(sys.argv[1])
    reference_points_num = (int)(sys.argv[2])
    ref_dirs = get_reference_directions("energy", objective_num, reference_points_num, seed = 1)
    num = 1
    for i in ref_dirs:
        for j in i:
            print(j)
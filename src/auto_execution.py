import subprocess

REPEAT_NUM = 3   # 반복 실험을 위한 횟수 (아직 미정)

MY_PROGRAM = "/home/jeus5771/Protein_NSGA3/src/Protein_NSGA3"
PROTEIN_FILE_PATH = [
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/Q5VZP5.fasta.txt", 
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/A4Y1B6.fasta.txt",
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/B3LS90.fasta.txt",
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/B4TWR7.fasta.txt",
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/Q91X51.fasta.txt",
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/Q89BP2.fasta.txt",
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/A6L9J9.fasta.txt",
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/Q88X33.fasta.txt",
    "/home/jeus5771/Protein_NSGA3/Protein_FASTA/B7KHU9.fasta.txt",
    ]
POPULATION_SIZE = ["100", "200", "400", "800"]
CYCLE_NUM = ["100", "200", "400", "800"]
CDS_NUM = ["2", "3", "4", "5", "6", "7", "8", "9", "10"]
MUTATION_PROB = ["0.05", "0.1", "0.15", "0.2", "0.25"]
OUTPUT_PATH = "/home/jeus5771/Protein_NSGA3/src/Output/"
OUTPUT_TXT = [
    "CUDA_Result_Q5VZP5.txt",
    "CUDA_Result_A4Y1B6.txt",
    "CUDA_Result_B3LS90.txt",
    "CUDA_Result_B4TWR7.txt",
    "CUDA_Result_Q91X51.txt",
    "CUDA_Result_Q89BP2.txt",
    "CUDA_Result_A6L9J9.txt",
    "CUDA_Result_Q88X33.txt",
    "CUDA_Result_B7KHU9.txt"
    ]


for protein_i in range(len(PROTEIN_FILE_PATH)):
    with open(OUTPUT_PATH + OUTPUT_TXT[protein_i], 'w') as file:
        for pop_size in POPULATION_SIZE:
            for cycles in CYCLE_NUM:
                for muation_prob in MUTATION_PROB:
                    for r in range(REPEAT_NUM):
                        command = [MY_PROGRAM] + [PROTEIN_FILE_PATH[protein_i]] + [pop_size] + [cycles] + [CDS_NUM[protein_i]] + [muation_prob]
    
                        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
                        file.write(f"N size : {pop_size}, Cycles : {cycles}, Mutation Probability : {muation_prob}")
                        if result.returncode == 0:
                            print(str(command) + " Complete")
                            file.write(result.stdout)
                        else:
                            print(str(command) + " Fail")
                            file.write(result.stderr)            
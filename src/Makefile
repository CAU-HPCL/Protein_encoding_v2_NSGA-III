NVCC := nvcc
NVCCFLAGS := -res-usage -use_fast_math --gpu-architecture=sm_89
SRCS := main.cu mutation.cu sorting.cu utils.cu
OBJS := $(SRCS:.cu=.o)
TARGET := Protein_NSGA3
# LDFLAGS := -lcudart
# LIBS := 

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $< -o $@ 

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
NVCC := nvcc
NVCCFLAGS := -res-usage -use_fast_math --gpu-architecture=sm_89
# LDFLAGS := -lcudart 
# LIBS := 
SRCS := main.cu
TARGET := Protein_NSGA3
OBJS := $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

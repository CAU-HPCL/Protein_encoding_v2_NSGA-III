NVCC := nvcc
NVCCFLAGS := -res-usage -use_fast_math --gpu-architecture=sm_89
LDFLAGS := -L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -lpython3.10
# LIBS := 
SRCS := main.cu
TARGET := Protein_NSGA3
OBJS := $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)

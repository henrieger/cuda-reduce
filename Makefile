CC = nvcc
CFLAGS = -I/usr/include/c++/10 -I/usr/lib/cuda/include/ --std=c++11
LDLIBS = -lm

EXEC = cudaReduceMax

all: $(EXEC)

cudaReduceMax: cudaReduceMax.cu
	$(CC) $(CFLAGS) cudaReduceMax.cu -o cudaReduceMax $(LDLIBS)

debug: CFLAGS += -I/usr/lib/nvidia-cuda-toolkit/compute-sanitizer -g -G
debug: all

clean:
	rm -r *.o
purge: clean
	rm -r $(EXEC)

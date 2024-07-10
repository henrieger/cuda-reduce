CC = nvcc
CFLAGS = -I/usr/include/c++/10 -I/usr/lib/cuda/include/ --std=c++11
LDLIBS = -lm

EXEC = cudaReduceMax

all: $(EXEC)

cudaReduceMax: cudaReduceMax.cu chrono.o
	$(CC) $(CFLAGS) cudaReduceMax.cu -o cudaReduceMax $(LDLIBS) chrono.o

chrono.o: chrono.c
	gcc -o chrono.o -c chrono.c

debug: CFLAGS += -I/usr/lib/nvidia-cuda-toolkit/compute-sanitizer -g -G
debug: all

clean:
	rm -r *.o
purge: clean
	rm -r $(EXEC)

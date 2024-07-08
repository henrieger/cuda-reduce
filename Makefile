CC = nvcc
CFLAGS = -I/usr/include/c++/10 -I/usr/lib/cuda/include/ --std=c++11
LDLIBS = -lm

EXEC = reduceMax

all: $(EXEC)

reduceMax: reduceMax.cu
	$(CC) $(CFLAGS) reduceMax.cu -o reduceMax $(LDLIBS)

debug: CFLAGS += -lineinfo -I/usr/lib/nvidia-cuda-toolkit/compute-sanitizer -g
debug: all

clean:
	rm -r *.o
purge: clean
	rm -r $(EXEC)

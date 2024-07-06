CC = nvcc
CFLAGS = -I/usr/local/cuda/cuda-samples/Common --threads 0 --std=c++11
LDLIBS = -lm

EXEC=reduceMax_persist reduceMax_atomic_persist

all: $(EXEC)

reduceMax_persist: reduce.cu
	$(CC) $(CFLAGS) reduce.cu -o reduceMax_persist $(LDLIBS)

reduceMax_persist: reduce.cu
	$(CC) $(CFLAGS) reduce.cu -o reduceMax_atomic_persist $(LDLIBS)

clean:
	rm -r *.o
purge: clean
	rm -r $(EXEC)

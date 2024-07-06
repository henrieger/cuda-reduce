CC = nvcc
CFLAGS = -I/usr/local/cuda/cuda-samples/Common --threads 0 --std=c++11
LDLIBS = -lm

EXEC = reduceMax

all: $(EXEC)

reduceMax: reduceMax.cu
	$(CC) $(CFLAGS) reduceMax.cu -o reduceMax $(LDLIBS)

clean:
	rm -r *.o
purge: clean
	rm -r $(EXEC)

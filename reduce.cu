#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#define SEED 31415926

#define GTX1080Ti 1
#define GTX1070 2
#define GPU GTX1080Ti

#if GPU == GTX1080Ti
    #define MP 28
    #define THREADS_PER_BLOCK 1024
    #define RESIDENT_BLOCKS_PER_MP 2
    #define NTA (MP*RESIDENT_BLOCKS_PER_MP*THREADS_PER_BLOCK)
#endif

#if GPU == GTX1070
    #define MP 28
    #define THREADS_PER_BLOCK 1024
    #define RESIDENT_BLOCKS_PER_MP 2
    #define NTA (MP*RESIDENT_BLOCKS_PER_MP*THREADS_PER_BLOCK)
#endif 

__global__ void reduce(const float *A, float *r) {
  return;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Uso: %s <TAM_VETOR>", argv[0]);
    exit(EXIT_FAILURE);
  }

  srand(SEED);

  int vectorSize = atoi(argv[1]);

  // Aloca espaço no host para vetor A e resultado
  float *h_A = (float *)malloc(vectorSize * sizeof(float));
  float h_result = 0;

  // Aloca espaço na GPU para vetor A e resultado
  float *d_A = NULL;
  float *d_result = NULL;
  cudaMalloc((void **)&d_A, vectorSize * sizeof(float));
  cudaMalloc((void **)&d_result, sizeof(float));

  // Inicializa vetor A com valores aleatórios entre 0 e 1
  for (int i = 0; i < vectorSize; i++)
    h_A[i] = (float)rand()/RAND_MAX;

  // Copia vetor A para GPU
  cudaMemcpy(h_A, d_A, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

  // Lança kernel
  // TODO

  // Copia resultado para o host
  cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);


  // Calcula redução normal em CPU
  float correct = 0;
  for (int i = 0; i < vectorSize; i++)
    correct = fmax(h_A[i], correct);

  // Checa corretude do resultado
  if (h_result != correct) {
    fprintf(stderr, "Resultado errado. Esperava %f e obteve %f", correct, h_result);
    exit(EXIT_FAILURE);
  }

  // Libera estruturas
  cudaFree(d_A);
  cudaFree(d_result);
  free(h_A);
}

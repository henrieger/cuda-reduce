#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define SEED 31415926

#define GTX1080Ti 1
#define GTX1070 2
#define GPU GTX1080Ti

#if GPU == GTX1080Ti
#define MP 28
#define THREADS_PER_BLOCK 1024
#define RESIDENT_BLOCKS_PER_MP 2
#endif

#if GPU == GTX1070
#define MP 28
#define THREADS_PER_BLOCK 1024
#define RESIDENT_BLOCKS_PER_MP 2
#endif

#define TOTAL_BLOCKS (MP * RESIDENT_BLOCKS_PER_MP)
#define NTA (TOTAL_BLOCKS * THREADS_PER_BLOCK)

__device__ float atomicMaxFloat(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void reduceMax_persist(float *max, float *Input,
                                  unsigned int nElements) {
  // Inicia vetor em shared memory com máximo de cada thread
  __shared__ float threadsMax[THREADS_PER_BLOCK];
  int t = threadIdx.x;
  threadsMax[t] = 0;

  // Reseta resposta
  if (t == 0)
    *max = 0;

  // FASE 1 - Computa o máximo para cada thread
  int initial = blockDim.x * blockIdx.x + t;
  for (int i = initial; i < nElements; i += NTA)
    threadsMax[t] = fmaxf(threadsMax[t], Input[i]);

  // FASE 2 - Computa o máximo do bloco usando o algoritmo dos slides
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (t < stride)
      threadsMax[t] = fmaxf(threadsMax[t], threadsMax[t + stride]);
  }

  // FASE 3 - Computa o máximo de todos os blocos usando atomicos
  if (t == 0)
    atomicMaxFloat(max, threadsMax[0]);
}

__global__ void reduceMax_atomic_persist(float *max, float *Input,
                                         unsigned int nElements) {
  return;
}

void errorAndAbort(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);

  printf("Abortado\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  cudaError_t err = cudaSuccess;

  if (argc != 3)
    errorAndAbort("Uso: %s <TAM_VETOR> <NR>\n", argv[0]);

  srand(SEED);

  int vectorSize = atoi(argv[1]);
  int repetitions = atoi(argv[2]);

  // Aloca espaço no host para vetor A e resultado
  float *h_A = (float *)malloc(vectorSize * sizeof(float));
  float h_max = 0;

  // Aloca espaço na GPU para vetor A e resultado
  float *d_A = NULL;
  float *d_max = NULL;
  err = cudaMalloc((void **)&d_A, vectorSize * sizeof(float));
  if (err != cudaSuccess)
    errorAndAbort("Erro ao alocar vetor A no dispositivo: %s\n",
                  cudaGetErrorString(err));
  err = cudaMalloc((void **)&d_max, sizeof(float));
  if (err != cudaSuccess)
    errorAndAbort("Erro ao alocar resultado no dispositivo: %s\n",
                  cudaGetErrorString(err));

  for (int i = 0; i < repetitions; i++) {
    // Inicializa vetor A com valores aleatórios
    for (int j = 0; j < vectorSize; j++) {
      float a = rand();
      float b = rand();
      h_A[j] = a * 100 + b;
    }

    // Copia vetor A para GPU
    err = cudaMemcpy(d_A, h_A, vectorSize * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      errorAndAbort("Erro ao copiar vetor A para dispositivo: %s\n",
                    cudaGetErrorString(err));

    // Wrapers do Thrust para vetor A
    thrust::device_ptr<float> thrust_A(d_A);
    thrust::device_vector<float> thrust_A_vector(thrust_A,
                                                 thrust_A + vectorSize);

    // Lança kernel
    reduceMax_persist<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>>(d_max, d_A,
                                                           vectorSize);
    err = cudaGetLastError();
    if (err != cudaSuccess)
      errorAndAbort("Erro ao lançar kernel reduceMax_persist: %s\n",
                    cudaGetErrorString(err));
    cudaDeviceSynchronize();

    // Copia resultado para o host
    err = cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      errorAndAbort("Erro ao copiar resultado para o host: %s\n",
                    cudaGetErrorString(err));

    // Calcula redução usando Thrust
    float correct =
        *thrust::max_element(thrust_A_vector.begin(), thrust_A_vector.end());
    printf("%f\n", correct);

    // Checa corretude do resultado
    if (h_max != correct)
      errorAndAbort("Resultado errado. Esperava %f e obteve %f\n", correct,
                    h_max);
  }

  // Libera estruturas
  cudaFree(d_A);
  cudaFree(d_max);
  free(h_A);
}
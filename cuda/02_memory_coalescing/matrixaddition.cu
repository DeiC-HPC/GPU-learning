#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

__global__ void matrixaddition(const int *a, const int *b, int *res,
                               ushort width, ushort height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < height) {
    for (int j = 0; j < width; j++) {
      res[i * width + j] = a[i * width + j] + b[i * width + j];
    }
  }
}

int main() {
  int height = 10000;
  int width = 10000;
  int blocksize = 1024;
  int dimx = ceil(((float)height) / blocksize);
  dim3 block(blocksize, 1, 1), grid(dimx, 1, 1);
  int memsize = width * height * sizeof(int);
  clock_t start, end;

  int *a, *b, *res;
  cudaMallocManaged(&a, memsize);
  cudaMallocManaged(&b, memsize);
  cudaMallocManaged(&res, memsize);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      a[i * width + j] = 1;
      b[i * width + j] = 1;
    }
  }

  start = clock();

  matrixaddition<<<grid, block>>>(a, b, res, width, height);
  cudaDeviceSynchronize();

  end = clock();

  printf("Elapsed time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

  bool allElementsAre2 = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (res[i * width + j] != 2) {
        allElementsAre2 = false;
      }
    }
  }

  if (allElementsAre2) {
    cout << "All numbers in matrix are 2" << endl;
  } else {
    cout << "Not all numbers in matrix are 2" << endl;
  }

  return 0;
}

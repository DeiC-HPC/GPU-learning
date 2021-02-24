#include "hip/hip_runtime.h"
#include <iostream>
#include <timer.h>

using namespace std;

__global__ void matrixaddition(const int *a, const int *b, int *res,
                               ushort width, ushort height) {
  /* ANCHOR: matrixaddition */
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < width) {
    for (int i = 0; i < height; i++) {
      res[i * width + j] = a[i * width + j] + b[i * width + j];
    }
  }
  /* ANCHOR_END: matrixaddition */
}

int main() {
  int height = 10000;
  int width = 10000;
  int blocksize = 1024;
  int dimx = ceil(((float)width) / blocksize);
  dim3 block(blocksize, 1, 1), grid(dimx, 1, 1);
  int memsize = width * height * sizeof(int);

  int *a, *b, *res;
  hipMallocManaged(&a, memsize);
  hipMallocManaged(&b, memsize);
  hipMallocManaged(&res, memsize);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      a[i * width + j] = 1;
      b[i * width + j] = 1;
    }
  }

  timer time;
  hipLaunchKernelGGL(matrixaddition, grid, block, 0, 0, a, b, res, width, height);
  hipDeviceSynchronize();
  cout << "Elapsed time: " << time.getTime() << endl;

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

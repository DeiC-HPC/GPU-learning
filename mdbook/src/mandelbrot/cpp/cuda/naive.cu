#include "cuComplex.h"
#include <fstream>
#include <iostream>
#include <timer.h>

#define T 32

using namespace std;

/* ANCHOR: mandelbrot */
__global__ void mandelbrot(const cuFloatComplex *zs, int *res, ushort width,
                           ushort height, ushort max_iterations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    cuFloatComplex z = zs[y * width + x];
    cuFloatComplex c = z;

    for (int i = 0; i < max_iterations; i++) {
      if (z.x * z.x + z.y * z.y <= 4.0f) {
        res[y * width + x] = i + 1;
        z = cuCmulf(z, z);
        z = cuCaddf(z, c);
      }
    }
  }
}
/* ANCHOR_END: mandelbrot */

int main() {
  int width = 1000;
  int height = 1000;
  int maxiterations = 100;
  float ymin = -2.0;
  float ymax = 2.0;
  float xmin = -2.5;
  float xmax = 1.5;
  int dimx = ceil(((float)width) / T);
  int dimy = ceil(((float)height) / T);
  dim3 block(T, T, 1), grid(dimx, dimy, 1);
  int resmemsize = width * height * sizeof(int);
  int cufloatmemsize = width * height * sizeof(cuFloatComplex);

  cuFloatComplex *zs = new cuFloatComplex[width * height];
  cuFloatComplex *zs_device;
  cudaMalloc((void **)&zs_device, cufloatmemsize);

  int *res = new int[width * height];
  int *res_device;
  cudaMalloc((void **)&res_device, resmemsize);

  timer time;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      zs[i * width + j] =
          make_cuFloatComplex(xmin + ((xmax - xmin) * j / (width - 1)),
                              ymax - ((ymax - ymin) * i / (height - 1)));
    }
  }

  cudaMemcpy(zs_device, zs, cufloatmemsize, cudaMemcpyHostToDevice);

  mandelbrot<<<grid, block>>>(zs_device, res_device, width, height,
                              maxiterations);
  cudaDeviceSynchronize();

  cudaMemcpy(res, res_device, resmemsize, cudaMemcpyDeviceToHost);
  cout << "Elapsed time: " << time.getTime() << endl;

  ofstream file;
  file.open("mandelbrot_naive.csv");

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (j != 0) {
        file << ",";
      }
      file << res[i * width + j];
    }
    file << endl;
  }

  file.close();

  return 0;
}

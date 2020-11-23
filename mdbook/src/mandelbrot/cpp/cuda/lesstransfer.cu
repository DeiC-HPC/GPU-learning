#include "cuComplex.h"
#include <fstream>
#include <iostream>
#include <timer.h>

#define T 32

using namespace std;

/* ANCHOR: mandelbrot */
__global__ void mandelbrot(const float *re, const float *im, int *res,
                           ushort width, ushort height, ushort max_iterations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    cuFloatComplex z = make_cuFloatComplex(re[x], im[y]);
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

  float *re = new float[width];
  float *im = new float[height];
  float *re_device, *im_device;
  cudaMalloc((void **)&re_device, width * sizeof(float));
  cudaMalloc((void **)&im_device, height * sizeof(float));

  int *res = new int[width * height];
  int *res_device;
  cudaMalloc((void **)&res_device, resmemsize);

  timer time;
  for (int i = 0; i < width; i++) {
    re[i] = xmin + ((xmax - xmin) * i / (width - 1));
  }
  for (int i = 0; i < height; i++) {
    im[i] = ymax - ((ymax - ymin) * i / (height - 1));
  }

  cudaMemcpy(re_device, re, width * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(im_device, im, height * sizeof(float), cudaMemcpyHostToDevice);

  mandelbrot<<<grid, block>>>(re_device, im_device, res_device, width, height,
                              maxiterations);
  cudaDeviceSynchronize();

  cudaMemcpy(res, res_device, resmemsize, cudaMemcpyDeviceToHost);
  cout << "Elapsed time: " << time.getTime() << endl;

  ofstream file;
  file.open("mandelbrot_lesstransfer.csv");

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

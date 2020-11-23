#include "cuComplex.h"
#include <fstream>
#include <iostream>
#include <timer.h>

#define T 32

using namespace std;

/* ANCHOR: mandelbrot */
__global__ void mandelbrot(int *res, ushort width, ushort height, float xmin,
                           float xmax, float ymin, float ymax,
                           ushort max_iterations) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float widthf = width - 1.0f;
  float heightf = height - 1.0f;

  if (x < width && y < height) {
    cuFloatComplex z =
        make_cuFloatComplex(xmin + ((xmax - xmin) * x / widthf),
                            ymax - ((ymax - ymin) * y / heightf));
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

  int *res = new int[width * height];
  int *res_device;
  cudaMalloc((void **)&res_device, resmemsize);

  timer time;
  mandelbrot<<<grid, block>>>(res_device, width, height, xmin, xmax, ymin, ymax,
                              maxiterations);
  cudaDeviceSynchronize();

  cudaMemcpy(res, res_device, resmemsize, cudaMemcpyDeviceToHost);
  cout << "Elapsed time: " << time.getTime() << endl;

  ofstream file;
  file.open("mandelbrot_gpuonly.csv");

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

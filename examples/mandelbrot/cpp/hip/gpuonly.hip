#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"
#include <fstream>
#include <iostream>
#include <timer.h>

#define T 32

using namespace std;

/* ANCHOR: mandelbrot */
__global__ void mandelbrot(
    int *res,
    ushort width,
    ushort height,
    float xmin,
    float xdelta,
    float ymin,
    float ydelta,
    int max_iterations) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  hipFloatComplex z = make_hipFloatComplex(xmin + x * xdelta, ymin + y * ydelta);
  hipFloatComplex c = z;

  int i;
  for (i = 0; i < max_iterations; i++) {
    if (z.x * z.x + z.y * z.y <= 4.0f) {
      z = hipCmulf(z, z);
      z = hipCaddf(z, c);
    } else {
      break;
    }
  }
  res[y * width + x] = i;
}
/* ANCHOR_END: mandelbrot */

int main() {
  int width = 100000;
  int height = 100000;
  int maxiterations = 100;
  float ymin = -2.0;
  float ymax = 2.0;
  float xmin = -2.5;
  float xmax = 1.5;

  int *res = new int[width * height];
  int *res_device;
  hipMalloc((void **)&res_device, width * height * sizeof(int));

  timer time;

  int dimx = ceil(((float)width) / T);
  int dimy = ceil(((float)height) / T);
  dim3 block(T, T, 1), grid(dimx, dimy, 1);

  hipLaunchKernelGGL(mandelbrot, grid, block, 0, 0,
      res_device,
      width,
      height,
      xmin, (xmax - xmin) / (width - 1.0),
      ymin, (ymax - ymin) / (height - 1.0),
      maxiterations
  );
  hipDeviceSynchronize();

  hipMemcpy(res, res_device, width * height * sizeof(int), hipMemcpyDeviceToHost);

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

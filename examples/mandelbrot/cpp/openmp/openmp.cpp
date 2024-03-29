#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <timer.h>

using namespace std;

/* ANCHOR: mandelbrot */
#pragma omp declare target
int mandelbrot(complex<float> z, size_t maxiterations) {
  complex<float> c = z;
  for (size_t i = 0; i < maxiterations; i++) {
    if (abs(z) > 2) {
      return i;
    }
    z = z * z + c;
  }

  return maxiterations;
}
#pragma omp end declare target
/* ANCHOR_END: mandelbrot */

int main() {
  size_t width = 50000;
  size_t height = 50000;
  int maxiterations = 100;
  float ymin = -2.0;
  float ymax = 2.0;
  float xmin = -2.5;
  float xmax = 1.5;

  complex<float> *zs = new complex<float>[width * height];

  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      zs[i * width + j] =
          complex<float>(
              xmin + ((xmax - xmin) * j / (width - 1)),
              ymin + ((ymax - ymin) * i / (height - 1))
          );
    }
  }

  int *res = new int[width * height];

  timer time;

/* ANCHOR: loops */
  #pragma omp target teams distribute parallel for collapse(2) map(to: zs[:width * height]) map(from: res[:width * height])
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      res[i * width + j] = mandelbrot(zs[i * width + j], maxiterations);
    }
  }
  /* ANCHOR_END: loops */

  cout << "Elapsed time: " << time.getTime() << endl;

  ofstream file;
  file.open("mandelbrot_openmp.csv");

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

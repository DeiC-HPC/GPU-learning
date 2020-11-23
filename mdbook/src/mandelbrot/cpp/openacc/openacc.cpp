#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <timer.h>

using namespace std;

/* ANCHOR: mandelbrot */
#pragma acc routine
int mandelbrot(complex<float> z, int maxiterations) {
  complex<float> c = z;
  for (int i = 0; i < maxiterations; i++) {
    if (abs(z) > 2) {
      return i;
    }
    z = z * z + c;
  }

  return maxiterations;
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
  int *res = new int[width * height];

  timer time;
#pragma acc parallel loop collapse(2) copyout(res[:width * height])
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      res[i * width + j] =
          mandelbrot(complex<float>(xmin + ((xmax - xmin) * j / (width - 1)),
                                    ymax - ((ymax - ymin) * i / (height - 1))),
                     maxiterations);
    }
  }
  cout << "Elapsed time: " << time.getTime() << endl;

  ofstream file;
  file.open("mandelbrot_openacc.csv");

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

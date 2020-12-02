#include <iostream>
#include <timer.h>

using namespace std;

int main() {
  int height = 10000;
  int width = 10000;

  int *a = new int[width * height];
  int *b = new int[width * height];
  int *res = new int[width * height];

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      a[i * width + j] = 1;
      b[i * width + j] = 1;
    }
  }

  timer time;
  /* ANCHOR: matrixaddition */
  #pragma omp target teams distribute parallel for \
      map(to: a[:width * height]) \
      map(to: b[:width * height]) \
      map(from: res[:width * height])
  for (int j = 0; j < width; j++) {
    for (int i = 0; i < height; i++) {
      res[i * width + j] = a[i * width + j] + b[i * width + j];
    }
  }
  /* ANCHOR_END: matrixaddition */
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

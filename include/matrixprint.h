#include <iostream>
#include <math.h>

using namespace std;

void printMatrix(int *matrix, int height, int width) {
  int logmax = 0;
  for (int i = 0; i < height; i++) {
    if (i == 3 && height > 10) {
      i += height - 6;
    }

    for (int j = 0; j < width; j++) {
      if (j == 3 && width > 10) {
        j += width - 6;
      }
      int num = matrix[i * width + j];
      int lognum = log10(abs(num));
      if (lognum > logmax) {
        logmax = lognum;
      }
    }
  }
  for (int i = 0; i < height; i++) {
    if (i == 3 && height > 10) {
      cout << " ... " << endl;
      i += height - 6;
    }
    for (int j = 0; j < width; j++) {
      if (j == 3 && width > 10) {
        cout << " ... ";
        j += width - 6;
      }
      int num = matrix[i * width + j];
      int lognum = log10(abs(num));
      if (lognum < logmax) {
        for (int k = 0; k < logmax - lognum; k++) {
          cout << " ";
        }
      }
      cout << num << " ";
    }
    cout << endl;
  }
}

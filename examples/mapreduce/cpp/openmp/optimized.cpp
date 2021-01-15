#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

int main() {
  int num = 100000000;
  int *elements = new int[num];
  long res = 0;
  clock_t start, end;

  start = clock();

  /* ANCHOR: mapreduce */
  #pragma omp target data map(alloc: elements[:num])
  {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < num; i++) {
      elements[i] = i;
    }

    #pragma omp target teams distribute parallel for reduction(+: res) map(from: res)
    for (int i = 0; i < num; i++) {
      res += elements[i];
    }
  }
  /* ANCHOR_END: mapreduce */

  end = clock();

  cout << "Elapsed time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

  cout << res << " " << elements[10000] << endl;
}

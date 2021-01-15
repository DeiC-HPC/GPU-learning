#include <iostream>
#include <timer.h>

using namespace std;

int main() {
  int num = 100000000;
  int *elements = new int[num];
  long res = 0;

  timer time;

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

  cout << "Elapsed time: " << time.getTime() << endl;

  cout << "The result is: " << res << endl;
}

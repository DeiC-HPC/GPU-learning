#include <iostream>
#include <stdlib.h>
#include <timer.h>

using namespace std;

int main() {
  int num = 100000000;
  int *elements = new int[num];
  long long res = 0;

  timer time;

  /* ANCHOR: mapreduce */
  #pragma acc data create(elements[:num])
  {
    #pragma acc parallel loop
    for (int i = 0; i < num; i++) {
      elements[i] = i;
    }
    #pragma acc parallel loop reduction(+ : res)
    for (int i = 0; i < num; i++) {
      res += elements[i];
    }
  }
  /* ANCHOR_END: mapreduce */

  cout << "Elapsed time: " << time.getTime() << endl;

  cout << "The result is: " << res << endl;
}

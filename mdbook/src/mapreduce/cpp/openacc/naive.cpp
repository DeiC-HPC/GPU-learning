#include <iostream>
#include <timer.h>

using namespace std;

int main() {
  int num = 100000000;
  int *elements = new int[num];
  long res = 0;

  timer time;

  #pragma acc parallel loop copy(elements[:num])
  for (int i = 0; i < num; i++) {
    elements[i] = i;
  }

  #pragma acc parallel loop reduction(+: res) copy(elements[:num])
  for (int i = 0; i < num; i++) {
    res += elements[i];
  }

  cout << "Elapsed time: " << time.getTime() << endl;

  cout << "The result is: " << res << endl;
}

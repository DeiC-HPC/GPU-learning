#include <iostream>
#include <timer.h>

using namespace std;

int main() {
  int num = 500;
  int memsize = num * num * num;
  long long *elements = new long long[memsize];

  #pragma acc data copyout(elements[:memsize])
  {
    #pragma acc parallel loop
    for (long long i = 0; i < memsize; i++) {
      elements[i] = i;
    }

    timer time;

    for (int j = 1; j < num; j++) {
      #pragma acc parallel loop collapse(2) present(elements[:memsize])
      for (int i = 0; i < num; i++) {
        for (int k = 0; k < num; k++) {
          elements[i * num * num + j * num + k] +=
              elements[i * num * num + (j - 1) * num + k];
        }
      }
    }

    cout << "Elapsed time: " << time.getTime() << endl;
  }

  cout << elements[memsize - 1] << endl;
}

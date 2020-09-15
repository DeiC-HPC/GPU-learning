#include<stdlib.h>
#include<iostream>
#include<time.h>

using namespace std;

int main() {
    int num = 100000000;
    int* elements = (int*)malloc(num*sizeof(num));
    long res = 0;
    clock_t start,end;

    start = clock();

    #pragma acc parallel loop copy(elements[:num])
    for (int i = 0; i < num; i++) {
        elements[i] = i;
    }

    #pragma acc parallel loop reduction(+:res) copy(elements[:num])
    for (int i = 0; i < num; i++) {
        res += elements[i];
    }

    end = clock();

    cout << "Elapsed time: " << (double)(end-start)/CLOCKS_PER_SEC << endl;

    cout << "The result is: " << res << endl;
}

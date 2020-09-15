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


    #pragma omp target data map(alloc:elements[:num]) map(from:res)
    {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num; i++) {
            elements[i] = i;
        }

        #pragma omp target teams distribute parallel for reduction(+:res)
        for (int i = 0; i < num; i++) {
            res += elements[i];
        }
    }

    end = clock();

    cout << "Elapsed time: " << (double)(end-start)/CLOCKS_PER_SEC << endl;

    cout << res << " " << elements[10000] << endl;
}

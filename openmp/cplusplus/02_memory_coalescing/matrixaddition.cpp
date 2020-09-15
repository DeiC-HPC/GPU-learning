#include<stdlib.h>
#include<iostream>
#include<time.h>

using namespace std;

int main() {
    int height = 10000;
    int width = 10000;
    int memsize = width*height*sizeof(int);
    clock_t start,end;

    int* a = (int*)malloc(memsize);
    int* b = (int*)malloc(memsize);
    int* res = (int*)malloc(memsize);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            a[i*width+j] = 1;
            b[i*width+j] = 1;
        }
    }

    start = clock();

    #pragma omp target teams distribute parallel for map(to:a[:width*height]) map(to:b[:width*height]) map(from:res[:width*height])
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            res[i*width+j] = a[i*width+j] + b[i*width+j];
        }
    }

    end = clock();

    printf("Elapsed time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    bool allElementsAre2 = true;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (res[i*width+j] != 2) {
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

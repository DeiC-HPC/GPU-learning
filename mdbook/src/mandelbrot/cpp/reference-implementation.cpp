#include<iostream>
#include<fstream>
#include<complex>
#include<cmath>
#include<time.h>

using namespace std;

/* ANCHOR: mandelbrot */
int mandelbrot(complex<float> z, int maxiterations) {
    complex<float> c = z;
    for (int i = 0; i < maxiterations; i++) {
        if (abs(z) > 2) {
            return i;
        }
        z = z*z + c;
    }

    return maxiterations;
}
/* ANCHOR_END: mandelbrot */

int main() {
    int width = 2000;
    int height = 2000;
    int maxiterations = 100;
    float ymin = -2.0;
    float ymax = 2.0;
    float xmin = -2.5;
    float xmax = 1.5;
    ofstream file;
    file.open("mandelbrot.csv");
    clock_t start,end;

    start = clock();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (j != 0) {
                file << ",";
            }
            file << mandelbrot(
                    complex<float>(
                        xmin + ((xmax-xmin)*j/(width-1)),
                        ymax - ((ymax-ymin)*i/(height-1))),
                    maxiterations);
        }
        file << endl;
    }

    file.close();
    end = clock();

    printf("Elapsed time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    return 0;
}

#include<iostream>
#include<fstream>
#include<complex>
#include<cmath>
#include<timer.h>

using namespace std;

#pragma omp declare target
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
#pragma omp end declare target

int main() {
    int width = 2000;
    int height = 2000;
    int maxiterations = 100;
    float ymin = -2.0;
    float ymax = 2.0;
    float xmin = -2.5;
    float xmax = 1.5;
    ofstream file;
    int* res = new int[width*height];

    timer time;
    #pragma omp target teams distribute parallel for collapse(2) map(from:res[:height*width])
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            res[i*width+j] = mandelbrot(
                    complex<float>(
                        xmin + ((xmax-xmin)*j/(width-1)),
                        ymax - ((ymax-ymin)*i/(height-1))),
                    maxiterations);
        }
    }
    cout << "Elapsed time: " << time.getTime() << endl;

    file.open("mandelbrot_openmp.csv");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (j != 0) {
                file << ",";
            }
            file << res[i*width+j];
        }
        file << endl;
    }

    file.close();

    return 0;
}

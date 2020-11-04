#include<stdlib.h>
#include<iostream>
#include<fstream>
#include "cuComplex.h"
#include<time.h>

#define T 32

using namespace std;

__global__ void mandelbrot(
    const float *re,
    const float *im,
    int *res,
    ushort width,
    ushort height,
    ushort max_iterations)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    if (x < width && y < height) {
        cuFloatComplex z = make_cuFloatComplex(re[x], im[y]);
        cuFloatComplex c = z;

        for (int i = 0; i < max_iterations; i++) {
            if (z.x*z.x + z.y*z.y <= 4.0f) {
                res[y*width+x] = i+1;
                z = cuCmulf(z, z);
                z = cuCaddf(z, c);
            }
        }
    }
}

int main() {
    int width = 1000;
    int height = 1000;
    int maxiterations = 100;
    float ymin = -2.0;
    float ymax = 2.0;
    float xmin = -2.5;
    float xmax = 1.5;
    int dimx = ceil(((float)width)/T);
    int dimy = ceil(((float)height)/T);
    dim3 block(T, T, 1), grid(dimx, dimy, 1);
    int resmemsize = width*height*sizeof(int);
    int floatmemsize = width*height*sizeof(float);
    clock_t start,end;

    start = clock();
    float* im = (float*)malloc(floatmemsize);
    float* re = (float*)malloc(floatmemsize);
    float *im_device, *re_device;
    cudaMalloc((void**)&im_device, floatmemsize);
    cudaMalloc((void**)&re_device, floatmemsize);

    int* res = (int*)malloc(resmemsize);
    int* res_device;
    cudaMalloc((void**)&res_device, resmemsize);

    for (int i = 0; i < width; i++) {
        re[i] = xmin + ((xmax-xmin)*i/(width-1));
    }
    for (int i = 0; i < height; i++) {
        im[i] = ymax - ((ymax-ymin)*i/(height-1));
    }

    cudaMemcpy(re_device, re, floatmemsize, cudaMemcpyHostToDevice);
    cudaMemcpy(im_device, im, floatmemsize, cudaMemcpyHostToDevice);

    mandelbrot<<<grid, block>>>(
            re_device,
            im_device,
            res_device,
            width,
            height,
            maxiterations);
    cudaDeviceSynchronize();

    cudaMemcpy(res, res_device, resmemsize, cudaMemcpyDeviceToHost);

    ofstream file;
    file.open("mandelbrot_lesstransfer.csv");

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
    end = clock();

    printf("Elapsed time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    return 0;
}

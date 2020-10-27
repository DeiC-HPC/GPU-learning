#include<stdlib.h>
#include "matrixprint.h"
#define T 16

using namespace std;


__global__ void matrixtranspose(
    const int *A,
    int *trA,
    int colsA,
    int rowsA)
{
    __shared__ int tile[T][T+1];

    int x = blockIdx.x * T + threadIdx.x;
    int y = blockIdx.y * T + threadIdx.y;
    if (y < rowsA && x < colsA) {
        tile[threadIdx.y][threadIdx.x] = A[y*colsA+x];
    }

    __syncthreads();

    x = blockIdx.y * T + threadIdx.x;
    y = blockIdx.x * T + threadIdx.y;
    if (y < rowsA && x < colsA) {
        trA[y*rowsA + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int height = 23000;
    int width = 23000;
    int dimx = ceil(((float)height)/T);
    int dimy = ceil(((float)width)/T);
    dim3 block(T, T, 1), grid(dimx, dimy, 1);
    int memsize = width*height*sizeof(int);
    clock_t start,end;

    int *a, *trA;
    cudaMallocManaged(&a, memsize);
    cudaMallocManaged(&trA, memsize);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            a[i*width+j] = i*width+j;
        }
    }

    matrixtranspose<<<grid, block>>>(
            a,
            trA,
            width,
            height);
    cudaDeviceSynchronize();

    start = clock();

    matrixtranspose<<<grid, block>>>(
            a,
            trA,
            width,
            height);
    cudaDeviceSynchronize();

    end = clock();

    printf("Elapsed time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    printMatrix(trA, height, width);

    return 0;
}

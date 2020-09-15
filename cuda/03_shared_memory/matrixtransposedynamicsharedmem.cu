#include<stdlib.h>
#include "matrixprint.h"
#define T 16

using namespace std;


__global__ void matrixtranspose(
    const int *A,
    int *trA,
    ushort colsA,
    ushort rowsA)
{
    extern __shared__ int tile[];
    int sharedIdx = threadIdx.y*T + threadIdx.x;
    int i = blockIdx.x*T + threadIdx.x;
    int j = blockIdx.y*T + threadIdx.y;
    if( j < colsA && i < rowsA ) {
        tile[sharedIdx] = A[i * colsA + j];
    }
    __syncthreads();
    i = blockIdx.y*T + threadIdx.x;
    j = blockIdx.x*T + threadIdx.y;
    if(j < rowsA && i < colsA) {
        sharedIdx = threadIdx.x*T + threadIdx.y;
        trA[i * rowsA + j] = tile[sharedIdx];
    }
}

int main() {
    int height = 20000;
    int width = 20000;
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

    start = clock();

    matrixtranspose<<<grid, block, T*T*sizeof(int)>>>(
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

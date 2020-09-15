import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math

width = 10000
height = 10000

a = np.arange(height * width).astype(np.int32)
a.shape = (height, width)

trA = np.empty((width, height)).astype(np.int32)

dim_size = 16
block_size = (dim_size,dim_size,1)
grid_size = (int(math.ceil(height / float(dim_size))),
             int(math.ceil(width / float(dim_size))))

mod = SourceModule("""
            #define T """ + str(dim_size) + """
            __global__ void matrixtranspose(
                const int *A,
                int *trA,
                ushort colsA,
                ushort rowsA)
            {
                __shared__ int tile[T][T+1];
                int tidx = threadIdx.x;
                int tidy = threadIdx.y;
                int i = blockIdx.x*T + tidx;
                int j = blockIdx.y*T + tidy;
                if(j < colsA && i < rowsA) {
                    tile[tidy][tidx] = A[i * colsA + j];
                }
                __syncthreads();
                i = blockIdx.y*T + threadIdx.x;
                j = blockIdx.x*T + threadIdx.y;
                if(j < rowsA && i < colsA) {
                    trA[i * rowsA + j] = tile[tidx][tidy];
                }
            }
            """)

matrixtranspose = mod.get_function("matrixtranspose")
matrixtranspose(
        cuda.In(a),
        cuda.Out(trA),
        np.uint16(width),
        np.uint16(height),
        block=block_size,
        grid=grid_size)

print(trA)

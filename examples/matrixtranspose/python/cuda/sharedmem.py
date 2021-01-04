#!/usr/bin/env python
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math
import time

width = 25000
height = 25000

a = np.arange(height * width).astype(np.int32)
a.shape = (height, width)

trA = np.empty((width, height)).astype(np.int32)

dim_size = 32
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
                    tile[tidy][tidx] = A[j * colsA + i];
                }
                __syncthreads();
                i = blockIdx.y*T + tidx;
                j = blockIdx.x*T + tidy;
                if(j < rowsA && i < colsA) {
                    trA[j * rowsA + i] = tile[tidx][tidy];
                }
            }
            """)

matrixtranspose = mod.get_function("matrixtranspose")
start_time = time.time()
matrixtranspose(
        cuda.In(a),
        cuda.Out(trA),
        np.uint16(width),
        np.uint16(height),
        block=block_size,
        grid=grid_size)
total_time_shared = time.time() - start_time

print(trA)

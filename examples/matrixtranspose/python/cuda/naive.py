import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math
import time

mod = SourceModule("""
        __global__ void matrixtranspose(
            const int *A,
            int *trA,
            ushort colsA,
            ushort rowsA)
        {
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            int j = blockIdx.y*blockDim.y + threadIdx.y;
            if( j < colsA && i < rowsA ) {
                trA[j * rowsA + i] = A[i * colsA + j];
            }
        }
        """)

width = 25000
height = 25000

a = np.arange(height * width).astype(np.int32)
a.shape = (height, width)

trA = np.empty((width, height)).astype(np.int32)

dim_size = 32
block_size = (dim_size,dim_size,1)
grid_size = (int(math.ceil(height / float(dim_size))),
             int(math.ceil(width / float(dim_size))))

matrixtranspose = mod.get_function("matrixtranspose")
start_time = time.time()
matrixtranspose(
        cuda.In(a),
        cuda.Out(trA),
        np.uint16(width),
        np.uint16(height),
        block=block_size,
        grid=grid_size)
total_time_transpose = time.time() - start_time

print(trA)

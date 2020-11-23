#!/usr/bin/env python
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math

mod = SourceModule("""
        __global__ void matrixaddition(
            const int *a,
            const int *b,
            int *res,
            ushort width,
            ushort height)
        {
            int i = blockIdx.x*blockDim.x+threadIdx.x;

            if (i < height) {
                for (int j = 0; j < width; j++) {
                    res[i*width+j] = a[i*width+j]+b[i*width+j];
                }
            }
        }
        """)

width = 10000
height = 10000

a = np.ones((height, width)).astype(np.int32)
b = np.ones((height, width)).astype(np.int32)

res = np.empty((height, width)).astype(np.int32)

dim_size = 1024
block_size = (dim_size,1,1)
grid_size = (int(math.ceil(height / float(dim_size))),1)

matrixaddition = mod.get_function("matrixaddition")
matrixaddition(
        cuda.In(a),
        cuda.In(b),
        cuda.Out(res),
        np.uint16(width),
        np.uint16(height),
        block=block_size,
        grid=grid_size)

print(res)

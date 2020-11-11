import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math

mod = SourceModule("""
        #include "cuComplex.h"

        __global__ void mandelbrot(
            const cuFloatComplex *zs,
            int *res,
            ushort width,
            ushort height,
            ushort max_iterations)
        {
            int x = blockIdx.x*blockDim.x+threadIdx.x;
            int y = blockIdx.y*blockDim.y+threadIdx.y;

            if (x < width && y < height) {
                cuFloatComplex z = zs[x*width+y];
                cuFloatComplex c = z;

                for (int i = 0; i < max_iterations; i++) {
                    if (z.x*z.x + z.y*z.y <= 4.0f) {
                        res[x*width+y] = i+1;
                        z = cuCmulf(z, z);
                        z = cuCaddf(z, c);
                    }
                }
            }
        }
        """)

width = 1000
height = 1000
max_iterations = 100
xmin = -2.5
xmax = 1.5
ymin = -2.0
ymax = 2.0
reals = np.linspace(xmin, xmax, width)
imaginaries = np.linspace(ymax, ymin, height) * 1j

zs = (reals+imaginaries[:, np.newaxis]).flatten().astype(np.complex64)

res = np.empty(width*height).astype(np.int32)

if width > 512:
    dim_size = 32
else:
    dim_size = 16
block_size = (dim_size,dim_size,1)

# Assuming width == height
grids_needed = int(math.ceil(width / float(dim_size)))
grid_size = (grids_needed, grids_needed)

mandelbrot = mod.get_function("mandelbrot")
mandelbrot(
        cuda.In(zs),
        cuda.Out(res),
        np.uint16(width),
        np.uint16(height),
        np.uint16(max_iterations),
        block=block_size,
        grid=grid_size)

# Setting shape of array to help displaying it
res.shape = (width, height)

# Displaying the Mandelbrot set
fig, ax = plt.subplots()

plt.imshow(res, interpolation='bicubic', cmap=plt.get_cmap("terrain"))
plt.axis("off")
plt.tight_layout()

plt.show()
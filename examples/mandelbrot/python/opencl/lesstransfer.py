#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
import time
# ANCHOR: import
import pyopencl as cl
# ANCHOR_END: import

# Getting context for running on the GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# ANCHOR: mandelbrot
prg = cl.Program(ctx, """
        float2 complex_product(float2 a, float2 b) {
            float2 res;
            res.x = a.x*b.x - a.y*b.y;
            res.y = a.x*b.y + b.x*a.y;
            return res;
        }

        __kernel void mandelbrot(
            __global const float *re,
            __global const float *im,
            __global int *res,
            ushort width,
            ushort max_iterations)
        {
            int x = get_global_id(0);
            int y = get_global_id(1);

            float2 z = (float2)(re[x], im[y]);
            float2 c = z;

            res[y*width+x] = 0;
            for (int i = 0; i < max_iterations; i++) {
                if (z.x*z.x + z.y*z.y <= 4.0f) {
                    res[y*width+x] = i+1;
                    z = complex_product(z,z);
                    z.x = z.x + c.x;
                    z.y = z.y + c.y;
                }
            }
        }
        """).build()
# ANCHOR_END: mandelbrot

width = 1000
height = 1000
max_iterations = 100
xmin = -2.5
xmax = 1.5
ymin = -2.0
ymax = 2.0

# Creates a list of equally distributed numbers
reals = np.linspace(xmin, xmax, width).astype(np.float32)
imaginaries = np.linspace(ymin, ymax, height).astype(np.float32)

start_time = time.time()

res = np.empty(width*height).astype(np.int32)

# Flags for memory on GPU
mf = cl.mem_flags

# Creating arrays on the GPU and transfering data
reals_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=reals)
imaginaries_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imaginaries)
res_dev = cl.Buffer(ctx, mf.WRITE_ONLY, size=res.nbytes)

# Running the kernel
prg.mandelbrot(
        queue,
        (height, width),
        None,
        reals_dev,
        imaginaries_dev,
        res_dev,
        np.uint16(width),
        np.uint16(max_iterations))

# Copying result from GPU to memory
cl.enqueue_copy(queue, res, res_dev).wait()

total_time = time.time() - start_time
print("Elapsed time:", total_time)

# Setting shape of array to help displaying it
res.shape = (width, height)

# Displaying the Mandelbrot set
fig, ax = plt.subplots()

ax.imshow(res, interpolation='bicubic', cmap=plt.get_cmap("terrain"))
plt.axis("off")
plt.tight_layout()

plt.show()

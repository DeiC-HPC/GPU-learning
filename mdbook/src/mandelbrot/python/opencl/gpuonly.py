#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

# Getting context for running on the GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
        float2 complex_product(float2 a, float2 b) {
            float2 res;
            res.x = a.x*b.x - a.y*b.y;
            res.y = a.x*b.y + b.x*a.y;
            return res;
        }

        __kernel void mandelbrot(
            __global int *res,
            ushort width,
            float xmin,
            float xmax,
            float ymin,
            float ymax,
            ushort max_iterations)
        {
            int x = get_global_id(1);
            int y = get_global_id(0);
            float widthf = width - 1.0f;

            float2 z;
            z.x = xmin + ((xmax-xmin)*x/widthf);
            z.y = ymax - ((ymax-ymin)*y/widthf);
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


width = 1000
height = 1000
max_iterations = 100
xmin = -2.5
xmax = 1.5
ymin = -2.0
ymax = 2.0
res = np.empty(width*height).astype(np.int32)

mf = cl.mem_flags
res_dev = cl.Buffer(ctx, mf.WRITE_ONLY, size=res.nbytes)

prg.mandelbrot(
        queue,
        (height, width),
        None,
        res_dev,
        np.uint16(width),
        np.float32(xmin),
        np.float32(xmax),
        np.float32(ymin),
        np.float32(ymax),
        np.uint16(max_iterations))

# Copying result from GPU to memory
cl.enqueue_copy(queue, res, res_dev).wait()

# Setting shape of array to help displaying it
res.shape = (width, height)

# Displaying the Mandelbrot set
fig, ax = plt.subplots()

plt.imshow(res, interpolation='bicubic', cmap=plt.get_cmap("terrain"))
plt.axis("off")
plt.tight_layout()

plt.show()

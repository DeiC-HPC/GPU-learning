import numpy as np
import pyopencl as cl
import time

# Getting context for running on the GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
__kernel void matrixaddition(
    __global const int *a,
    __global const int *b,
    __global int *res,
    ushort width,
    ushort height)
{
    /* ANCHOR: matrixaddition */
    int j = get_global_id(0);

    for (int i = 0; i < height; i++) {
        res[i * width + j] = a[i * width + j] + b[i * width + j];
    }
    /* ANCHOR_END: matrixaddition */
}
""").build()

width = 10000
height = 10000

a = np.ones((height, width)).astype(np.int32)
b = np.ones((height, width)).astype(np.int32)

res = np.empty((height, width)).astype(np.int32)

mf = cl.mem_flags
a_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
res_dev = cl.Buffer(ctx, mf.WRITE_ONLY, size=res.nbytes)

start_time = time.time()

prg.matrixaddition(
        queue,
        (width,),
        None,
        a_dev,
        b_dev,
        res_dev,
        np.uint16(width),
        np.uint16(height))

cl.enqueue_copy(queue, res, res_dev).wait()

total_time = time.time() - start_time
print("Elapsed time:", total_time)

print(res)

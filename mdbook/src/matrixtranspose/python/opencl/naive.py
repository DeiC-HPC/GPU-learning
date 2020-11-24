#!/usr/bin/env python
import numpy as np
import pyopencl as cl
import time

# Getting context for running on the GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
        __kernel void matrixtranspose(
            __global const int *a,
            __global int *trA,
            ushort columnsA,
            ushort rowsA)
        {
            int i = get_global_id(0);
            int j = get_global_id(1);

            trA[j*rowsA+i] = a[i*columnsA+j];
        }
        """).build()

width = 25000
height = 25000

a = np.arange(height * width).astype(np.int32)
a.shape = (height, width)

trA = np.empty((width, height)).astype(np.int32)

mf = cl.mem_flags
a_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
trA_dev = cl.Buffer(ctx, mf.WRITE_ONLY, size=trA.nbytes)

start_time = time.time()
prg.matrixtranspose(
        queue,
        (height,width),
        None,
        a_dev,
        trA_dev,
        np.uint16(width),
        np.uint16(height))

cl.enqueue_copy(queue, trA, trA_dev).wait()
total_time_normal = time.time() - start_time

print(trA)

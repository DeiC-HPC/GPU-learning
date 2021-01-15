import numpy as np
import pyopencl as cl
import math
import time

# Getting context for running on the GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

width = 25000
height = 25000

dim_size = 32
grid_size = (dim_size * int(math.ceil(height/float(dim_size))),
             dim_size * int(math.ceil(width/float(dim_size))))

prg = cl.Program(ctx, """
            #define T """ + str(dim_size) + """
            __kernel void matrixtranspose(
                __global const int *a,
                __global int *trA,
                ushort columnsA,
                ushort rowsA)
            {
                int i = get_global_id(0);
                int j = get_global_id(1);

                int loc_i = get_local_id(0);
                int loc_j = get_local_id(1);

                __local int tile[T][T];

                if (i < rowsA && j < columnsA) {
                    tile[loc_j][loc_i] = a[i*columnsA+j];
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                int group_i = get_group_id(0);
                int group_j = get_group_id(1);

                i = group_j*T+loc_i;
                j = group_i*T+loc_j;
                if (i < columnsA && j < rowsA) {
                    trA[i*rowsA+j] =  tile[loc_i][loc_j];
                }
            }
            """).build()

a = np.arange(height * width).astype(np.int32)
a.shape = (height, width)

trA = np.empty((width, height)).astype(np.int32)

mf = cl.mem_flags
a_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
trA_dev = cl.Buffer(ctx, mf.WRITE_ONLY, size=trA.nbytes)

start_time = time.time()
prg.matrixtranspose(
        queue,
        grid_size,
        (dim_size,dim_size),
        a_dev,
        trA_dev,
        np.uint16(width),
        np.uint16(height))

cl.enqueue_copy(queue, trA, trA_dev).wait()
total_time_shared = time.time() - start_time

print(trA)

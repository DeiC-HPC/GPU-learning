Shared Memory
=============
In this part we are going to talk about shared memory and how we can use it to
improve performance.

0 In general
------------
Shared memory is a small portion of memory connected to each thread block, which
is controllable by the user. It is used to lessen the amount of reads and writes
to global memory as the latency is around 100 times lower. Also in cases where
you have many local variables, it can also be an advantage to use shared memory
as they could be pushed to global memory.

To use shared memory, you have to mark your variable with `__local`, like so
`__local int array[10][10];`. When using arrays it is also worth noting that
they can not be allocated with a dynamic size. This means that the size can not
come from a variable and must be written in as a part of the kernel code.

1 Matrix transposition
----------------------
In this section we will be looking at matrix transposition. It is a problem
where we will get problems with memory coalescing without using shared memory.
Our first implementation will just be the naïve one, where we will transpose
directly.

```c++
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
```

As we can see, we will not get a coalesced memory access when writing to global
memory.

2 Shared Memory implementation
------------------------------
To get coalesced memory access, we will use shared memory. We will use the
shared memory to save the part of global memory, which is read by the thread
block. We can then use this saved to write to the correct place in another
thread so we get coalesced access.

To help us do this, we have to look closer at some features in OpenCL. Firstly
we need to understand barriers. A barrier is a way to make sure that all threads
are synchronised. It works by stopping threads continuing until all threads
within the thread block have reached the barrier. In OpenCL, we can use barriers
in two different scenarios by changing the argument when calling `barrier`
function. `CLK_LOCAL_MEM_FENCE` waits until shared memory has been flushed and
`CLK_GLOBAL_MEM_FENCE` waits until global memory has been flushed. It is also
important to note, that all code must pass the same barrier at the same time.
Using barriers in a different way will result in undefined behaviour.
![One thread is yet to reach the barrier, so the two others are waiting](barrier.png)
![All threads have reached the barrier, so they now can continue](barrierdone.png)

Two additional things we need are `get_local_id` and `get_group_id`. These two
functions works like `get_global_id`. `get_local_id` gets the current index in
the thread block, we are working in, and `get_group_id` gets the index of the
thread block. We need these to take advantage shared memory, because the entire
thread block needs to swap group id dimension 0 and 1, and then we can transpose
inside the thread block.
![Swapping two thread blocks in a small grid](threadblocks.png)

```c++
    #define T 16
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
```

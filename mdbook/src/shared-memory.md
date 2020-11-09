# Shared Memory
{:.cpp-openmp f90-openmp cpp-openacc f90-openacc}
This concept is not included in this platform.

In this part we are going to talk about shared memory and how we can use it to
improve performance.

0 In general
------------
Shared memory is a small portion of memory connected to each thread block, which
is controllable by the user. It is used to lessen the amount of reads and writes
to global memory as the latency is around 100 times lower. Also in cases where
you have many local variables, it can also be an advantage to use shared memory
as they could be pushed to global memory.

To use shared memory, you have to mark your variable with `__shared__`, like so
`__shared__ int array[10][10];`. Shared memory can also be allocated dynamically
using the `extern` keyword. But you then have to add an extra argument, when
running the kernel to define how much shared memory you need. This is done using
a named argument called `shared`, where you define how many bytes of shared
memory you need.

1 Matrix transposition
----------------------
In this section we will be looking at matrix transposition. It is a problem
where we will get problems with memory coalescing without using shared memory.
Our first implementation will just be the naïve one, where we will transpose
directly.

```c++
__global__ void matrixtranspose(
    const int *A,
    int *trA,
    ushort colsA,
    ushort rowsA)
{
    int i = blockIdx.x*T + threadIdx.x;
    int j = blockIdx.y*T + threadIdx.y;
    if( j < colsA && i < rowsA ) {
        trA[j * rowsA + i] = A[i * colsA + j];
    }
}
```

As we can see, we will not get a coalesced memory access when writing to global
memory.

2 Shared Memory implementation
------------------------------
To get coalesced memory access, we will use shared memory. We will use the
shared memory to save the part of global memory, which is read by the thread
block. We can then use this saved to write to the correct place in another
thread to get coalesced access.

Before we go on, we have to introduce barriers. Barriers are a way to ensure
that all threads, within a thread block, have reached a specific point before
any can continue. This is useful especially when using larger kernels or shared
memory, where we can make sure that every thread in the thread block has come
beyond a specific point. In CUDA we can use a barrier by calling the
`__syncthreads()` function. It is also important to note, that all code must
pass the same barrier at the same time. Using barriers in a different way will
result in undefined behaviour.
![One thread is yet to reach the barrier, so the two others are waiting](barrier.png)
![All threads have reached the barrier, so they now can continue](barrier.png)

To get coalesced access with share memory, we need to use the `blockIdx` to move
our thread blocks. By swapping `blockIdx.x` and `blockIdx.y`, when we calculate
our position, we can simply transpose within the block in shared memory and
write that result to global memory.
![Swapping two thread blocks in a small grid](threadblocks.png)

```c++
#define T 16
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
        tile[tidy][tidx] = A[i * colsA + j];
    }
    __syncthreads();
    i = blockIdx.y*T + threadIdx.x;
    j = blockIdx.x*T + threadIdx.y;
    if(j < colsA && i < rowsA) {
        trA[i * rowsA + j] = tile[tidx][tidy];
    }
}
```

3 Dynamically allocated shared memory implementation
----------------------------------------------------
In this implementation we will use dynamically allocated shared memory instead
of allocating it directly in the kernel. It does not yield any specific
performance benefit to dynamically allocate shared memory. But it will make the
kernel more general and you will need less code to handle changing block sizes.
We will also need to change the way we call our kernel by adding a third
argument, defining the number of bytes needed, to the brackets in the function
call.
```c++
matrixtranspose<<<grid, block, T*T*sizeof(int)>>>
```

```c++
__global__ void matrixtranspose(
    const int *A,
    int *trA,
    ushort colsA,
    ushort rowsA)
{
    extern __shared__ int tile[];
    int sharedIdx = threadIdx.y*blockDim.y + threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if( j < colsA && i < rowsA ) {
        tile[sharedIdx] = A[i * colsA + j];
    }
    __syncthreads();
    i = blockIdx.y*blockDim.y + threadIdx.x;
    j = blockIdx.x*blockDim.x + threadIdx.y;
    if(j < colsA && i < rowsA) {
        sharedIdx = threadIdx.x*blockDim.x + threadIdx.y;
        trA[i * rowsA + j] = tile[sharedIdx];
    }
}
```

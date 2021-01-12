# Shared Memory
{:.code-info cpp-openmp f90-openmp cpp-openacc f90-openacc}
This concept is not included in this environment.

In this part we are going to talk about shared memory and how we can use it to
improve performance.

0 In general
------------
Shared memory is a small portion, typically in the order of tens of kilobytes,
of memory connected to each thread block, which is controllable by the user.
It is used to lessen the amount of reads and writes to global memory as the
latency is around 100 times lower. Also in cases where you have many local
variables, it can also be an advantage to use shared memory as they could be
pushed to global memory.

{:.code-info pycuda .cuda}
To use shared memory, you have to mark your variable with `__shared__`, like so
`__shared__ int array[10][10];`. Shared memory can also be allocated dynamically
using the `extern` keyword. But you then have to add an extra argument, when
running the kernel to define how much shared memory you need. This is done using
a named argument called `shared`, where you define how many bytes of shared
memory you need.

{:.code-info pyopencl}
To use shared memory, you have to mark your variable with `__local`, like so
`__local int array[10][10];`. When using arrays it is also worth noting that
they can not be allocated with a dynamic size. This means that the size can not
come from a variable and must be written in as a part of the kernel code.

1 Gaussian blur
---------------
In this section we will be looking at gaussian blur. It is a problem where all
elements are accessed many times in memory. Our first will just be the naïve one,
where we do not use shared memory.

{:.code cuda}
```c++
{{#include ../../examples/blur/cpp/cuda/naive.cu:gaussianblur}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/blur/cpp/cuda/naive.ipynb)

{:.code pycuda}
```c++
TODO: Make code
```
{:.code pyopencl}
```c++
TODO: Make code
```

Here we can see that every thread accesses many elements around itself depending
on `FILTER_SIZE`, which is `21` in our example.

2 Shared Memory implementation
------------------------------
To reduce accesses to global memory, we will use shared memory. We will use the
shared memory to save the part of global memory, which is read by the thread
block.

{:.code-info pycuda cuda}
Before we go on, we have to introduce barriers. Barriers are a way to ensure
that all threads, within a thread block, have reached a specific point before
any can continue. This is useful especially when using larger kernels or shared
memory, where we can make sure that every thread in the thread block has come
beyond a specific point. In CUDA we can use a barrier by calling the
`__syncthreads()` function. It is also important to note, that all code must
pass the same barrier at the same time. Using barriers in a different way will
result in undefined behaviour.

{:.code-info pyopencl}
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

{:.code-info pyopencl}
Two additional functions we need are `get_local_id` and `get_group_id`. These two
functions works like `get_global_id`. `get_local_id` gets the current index in
the thread block, we are working in, and `get_group_id` gets the index of the
thread block. We need these to take advantage shared memory, because the entire
thread block needs to swap group id dimension 0 and 1, and then we can transpose
inside the thread block.

{:.code cuda}
```c++
{{#include ../../examples/blur/cpp/cuda/shared_memory.cu:gaussianblur}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/blur/cpp/cuda/shared_memory.ipynb)

{:.code pycuda}
```c++
TODO: Make code
```
{:.code pyopencl}
```c++
TODO: Make code
```

3 Dynamically allocated shared memory implementation
----------------------------------------------------
{:.code-info pyopencl}
This feature does not exist in OpenCL

In this implementation we will use dynamically allocated shared memory instead
of allocating it directly in the kernel. It does not yield any specific
performance benefit to dynamically allocate shared memory. But it will make it
possible to use it for multiple purposes or with changing block sizes.

{:.code-info cuda}
We will also need to change the way we call our kernel by adding a third
argument, defining the number of bytes needed, to the brackets in the function
call.

{:.code cuda}
```c++
{{#include ../../examples/blur/cpp/cuda/dynamic_shared_memory.cu:call}}
```

{:.code-info pycuda}
We will also need to change the way we call our kernel by adding the argument
`shared`, defining the number of bytes needed, to the function call.

{:.code cuda}
```c++
{{#include ../../examples/blur/cpp/cuda/dynamic_shared_memory.cu:gaussianblur}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/blur/cpp/cuda/dynamic_shared_memory.ipynb)

{:.code pycuda}
```c++
TODO: Make code
```
{:.code pyopencl}
```c++
TODO: Make code
```

# Shared Memory

In this part we are going to talk about shared memory and how we can use it to
improve performance.

In general
----------
Shared memory is a small portion, typically in the order of tens of kilobytes,
of memory connected to each thread block, which is controllable by the user.
It is used to lessen the amount of reads and writes to global memory as the
latency is around 100 times lower. Also in cases where you have many local
variables, it can also be an advantage to use shared memory as they could be
pushed to global memory.

To use shared memory, you have to mark your variable with `__shared__`, like so
`__shared__ int array[10][10];`. Shared memory can also be allocated dynamically
using the `extern` keyword. But you then have to add an extra argument, when
running the kernel to define how much shared memory you need. This is done using
a named argument called `shared`, where you define how many bytes of shared
memory you need.

Gaussian blur
-------------
In this section we will be looking at gaussian blur. It is a problem where all
elements are accessed many times in memory. Our first will just be the naïve one,
where we do not use shared memory.

=== "CUDA"
    ```c++ linenums="1"
    --8<-- "../examples/blur/cpp/cuda/naive.cu:17:34"
    ```

    [Run the code in Jupyter](/jupyter/lab/tree/blur/cpp/cuda/naive.ipynb)

=== "HIP"
    ```c++ linenums="1"
    --8<-- "../examples/blur/cpp/hip/naive.hip:18:35"
    ```

    [Run the code in Jupyter](/jupyter/lab/tree/blur/hip/cuda/naive.ipynb)


Here we can see that every thread accesses many elements around itself depending
on `FILTER_SIZE`, which is `21` in our example.

Shared Memory implementation
----------------------------
To reduce accesses to global memory, we will use shared memory. We will use the
shared memory to save the part of global memory, which is read by the thread
block.

Before we go on, we have to introduce barriers. Barriers are a way to ensure
that all threads, within a thread block, have reached a specific point before
any can continue. This is useful especially when using larger kernels or shared
memory, where we can make sure that every thread in the thread block has come
beyond a specific point. In CUDA and HIP we can use a barrier by calling the
`__syncthreads()` function. It is also important to note, that all code must
pass the same barrier at the same time. Using barriers in a different way will
result in undefined behaviour.

<figure markdown>
  ![One thread is yet to reach the barrier, so the two others are waiting](barrier.png)
  <figcaption>One thread is yet to reach the barrier, so the two others are waiting</figcaption>
</figure>

<figure markdown>
  ![All threads have reached the barrier, so they now can continue](barrierdone.png)
  <figcaption>All threads have reached the barrier, so they now can continue</figcaption>
</figure>

=== "CUDA"
    ```c++ linenums="1"
    --8<-- "../examples/blur/cpp/cuda/shared_memory.cu:15:53"
    ```

    [Run the code in Jupyter](/jupyter/lab/tree/blur/cpp/cuda/naive.ipynb)

=== "HIP"
    ```c++ linenums="1"
    --8<-- "../examples/blur/cpp/hip/shared_memory.hip:16:54"
    ```

    [Run the code in Jupyter](/jupyter/lab/tree/blur/hip/cuda/naive.ipynb)


Dynamically allocated shared memory implementation
--------------------------------------------------

In this implementation we will use dynamically allocated shared memory instead
of allocating it directly in the kernel. It does not yield any specific
performance benefit to dynamically allocate shared memory. But it will make it
possible to use it for multiple purposes or with changing block sizes.

=== "CUDA"
    We will also need to change the way we call our kernel by adding a third
    argument, defining the number of bytes needed, to the brackets in the function
    call.

    ```c++ linenums="1"
    --8<-- "../examples/blur/cpp/cuda/dynamic_shared_memory.cu:76:77"
    ```

    ```c++ linenums="1"
    --8<-- "../examples/blur/cpp/cuda/dynamic_shared_memory.cu:17:54"
    ```

    [Run the code in Jupyter](/jupyter/lab/tree/blur/cpp/cuda/dynamic_shared_memory.ipynb)

=== "HIP"
    We will also need to change the way we call our kernel by adding a third
    argument, defining the number of bytes needed, to the brackets in the function
    call.

    ```c++ linenums="1"
    --8<-- "../examples/blur/cpp/hip/dynamic_shared_memory.hip:77:78"
    ```

    ```c++ linenums="1"
    --8<-- "../examples/blur/cpp/hip/dynamic_shared_memory.hip:18:55"
    ```

    [Run the code in Jupyter](/jupyter/lab/tree/blur/cpp/cuda/dynamic_shared_memory.ipynb)


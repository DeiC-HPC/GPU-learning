# Native

{:.code-info cpp-openmp cpp-openacc f90-openmp f90-openacc}
This part of the book is not relevant for you chosen environment. Please go
[here](./directives.md) or change environment.

$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$, \(p(x|y) = \frac{p(y|x)p(x)}{p(y)}\)
--8<-- "docs/mandelbrot-sequential-implementation.md"

How do we then use the GPU?
---------------------------
Writing
**TODO**: Some intro text

Kernels, as the functions running on GPUs are called, have `__global__` before
their return type and name. The return type will always be `void` because these
functions does not return anything. Instead the data is copied to and from the
GPU.

```c++
__global__ void someKernel(
    const float *readOnlyArgument,
    float *writableArgument,
    float someConstant)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
}
```

With the code above we see three variables, `blockIdx`, `blockDim`, and `threadIdx`,
that we have not defined. These will be instantiated when we are running and
tell us know where we are running. When running our code, it is run in a grid of
thread block. Each thread block can have up to 1024 threads and must be a power
of 2.

=== "CUDA"
    So how do we allocate memory on the GPU and copy data to and from it? There are
    two ways that you can do this. Firstly you can use `cudaMalloc`, where you have
    control and choose when to copy to and from the GPU.

    ```c++
    float* numbers = (float*)malloc(n*sizeof(float));
    float* numbers_device;
    cudaMalloc((void**)&numbers_device, n*sizeof(float));

    // Copying to the device
    cudaMemcpy(numbers_device, numbers, n*sizeof(float), cudaMemcpyHostToDevice);
    // Copying from the device
    cudaMemcpy(numbers, numbers_device, n*sizeof(float), cudaMemcpyDeviceToHost);

    ```

    It is also possible to use `cudaMallocManaged`, where copying will be done for
    you. This can lead to worse performance.

    ```c++
    float* someMem;
    cudaMallocManaged(&someMem, n*sizeof(float));
    ```

=== "HIP"
    So how do we allocate memory on the GPU and copy data to and from it? There are
    two ways that you can do this. Firstly you can use `hipMalloc`, where you have
    control and choose when to copy to and from the GPU.

    ```c++
    float* numbers = (float*)malloc(n*sizeof(float));
    float* numbers_device;
    hipMalloc((void**)&numbers_device, n*sizeof(float));

    // Copying to the device
    hipMemcpy(numbers_device, numbers, n*sizeof(float), hipMemcpyHostToDevice);
    // Copying from the device
    hipMemcpy(numbers, numbers_device, n*sizeof(float), hipMemcpyDeviceToHost);
    ```
    It is also possible to use `hipMallocManaged`, where copying will be done for
    you. This can lead to worse performance. Given it is not supported on all GPUs,
    we should also include a compatibility check before making the calls.

    ```c++
    int p_gpuDevice = 0; // GPU device number
    int managed_memory = 0;
    hipGetDeviceAttribute(&managed_memory, hipDeviceAttributeManagedMemory,p_gpuDevice));
    float* someMem;
    if (!managed_memory) {
        // Return an error message
    } else {
        hipMallocManaged(&someMem, n*sizeof(float));
    }
    ```

When calling our kernel we need to define the dimensions of our kernel and
thread blocks. As said earlier the have up to 1024 threads and each dimension
must be a power of 2. The grid is then defined as the number of thread blocks we
want in each dimension.

```c++
dim3 grid(n,m,1);
dim3 block(16,16,1);

someKernel<<< grid, block >>>(readable, writable, 5.0f);
```

Naïve implementation
--------------------
=== "CUDA"
    In this version we have taken the naïve approach and done a direct translation
    of the program. To use the library for complex arithmetic, we start by writing
    `#include "cuComplex.h"`. This enables us to use the type `cuFloatComplex`, and
    the functions `cuCmulf` (multiplication of complex numbers) and `cuCaddf`
    (addition of complex numbers). Underneath the `cuFloatComplex` type is a vector
    type called `float2`, a 2D floating point vector. CUDA has multiple types, which
    support vector types, those are char, uchar, short, ushort, int, uint,
    long, ulong, longlong, ulonglong, float, and double. The length of the vector
    types can be 2, 3, and 4.

=== "HIP"
    In this version we have taken the naïve approach and done a direct translation
    of the program. To use the library for complex arithmetic, we start by writing
    `#include "hip/hip_complex.h"`. This enables us to use the type `hipFloatComplex`, and
    the functions `hipCmulf` (multiplication of complex numbers) and `hipCaddf`
    (addition of complex numbers). Underneath the `hipFloatComplex` type is a vector
    type called `float2`, a 2D floating point vector. HIP has multiple types, which
    support vector types, those are char, uchar, short, ushort, int, uint,
    long, ulong, longlong, ulonglong, float, and double. The length of the vector
    types can be 2, 3, and 4.

The only translation we have done in this version is the `mandelbrot`
function and the complex arithmetic, which means all data is still generated and
sent from the host. But looking at the function we see, that we have to send the
width and height to the function is because we are running in thread blocks, as
described earlier. We could end up out of bounds of our array, which we do not
want and therefore we have this `if`-statement.

=== "CUDA"
    ```c++
    --8<-- "../examples/mandelbrot/cpp/cuda/naive.cu:11:37"
    ```
    [Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/cpp/cuda/naive.ipynb)

=== "HIP"
    ```c++
    --8<-- "../examples/mandelbrot/cpp/hip/naive.hip:12:37"
    ```


Less transfer implementation
----------------------------
Transferring data to and from the GPU takes time, which in turn makes our
calculations slower. So we should try to limit how much data we move around.
In the naïve version we generate our data on the host and create a 1000 by 1000
matrix, which is then transfered to the GPU. But we can be smarter than that.
By sending our lists of real and imaginary parts, we can then combine them on
the GPU saving both time and space, because we already have the coordinates of
from our two global ids.

=== "CUDA"
    ```c++
    --8<-- "../examples/mandelbrot/cpp/cuda/lesstransfer.cu:11:38"
    ```

=== "HIP"
    ```c++
    --8<-- "../examples/mandelbrot/cpp/hip/lesstransfer.hip:12:39"
    ```

GPU only implementation
-----------------------
But we can also eliminate the need for transfering data to GPU completely by
letting the GPU do it. This will reduce total computation time
considerably, especially when calculating with a higher resolution. Of course we
still need to transfer the result array from the GPU, which is the majority of
our data transfer, but reducing data transfer should be a priority.

=== "CUDA"
    ```c++
    --8<-- "../examples/mandelbrot/cpp/cuda/gpuonly.cu:11:40"
    ```

=== "HIP"
    ```c++
    --8<-- "../examples/mandelbrot/cpp/hip/gpuonly.hip:12:41"
    ```
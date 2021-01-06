# Native

{:.code-info cpp-openmp cpp-openacc f90-openmp f90-openacc}
This part of the book is not relevant for you chosen environment. Please go
[here](./directives.md) or change environment.

{{#include mandelbrot-sequential-implementation.md}}

1 How do we then use the GPU?
-----------------------------
**TODO**: Some intro text

{:.code-info pycuda pyopencl}
**TODO**: Some text about importing modules

{:.code cuda}
```cpp
{{#include ../../examples/mandelbrot/cpp/cuda/naive.cu:import}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/cpp/cuda/naive.ipynb)

{:.code pycuda}
```python
{{#include ../../examples/mandelbrot/python/cuda/naive.py:import}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/python/cuda/naive.ipynb)

{:.code pyopencl}
```python
{{#include ../../examples/mandelbrot/python/opencl/naive.py:import}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/python/opencl/naive.ipynb)

{:.code-info cuda pycuda}
Kernels, as the functions running on GPUs are called, have `__global__` before
their return type and name. The return type will always be `void` because these
functions does not return anything. Instead the data is copied to and from the
GPU.

{:.code cuda pycuda}
```c++
__global__ void someKernel(
    const float *readOnlyArgument,
    float *writableArgument,
    float someConstant)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
}
```

{:.code-info cuda pycuda}
With the code above we see three variables, blockIdx, blockDim, and threadIdx,
that we have not defined. These will be instantiated when we are running and
tell us know where we are running. When running our code, it is run in a grid of
thread block. Each thread block can have up to 1024 threads and must be a power
of 2.

{:.code-info pyopencl}
Kernels, as the functions running on GPUs are called, have `__kernel` before
their return type and name. The return type will always be `void` because these
functions does not return anything. Instead the data is copied to and from the
GPU. The kernels are compiled during runtime in OpenCL where they are loaded from
a string. This can of course be loaded from a file but in our examples we have
chosen not to as to keep the examples simpler.

{:.code pyopencl}
```python
prg = cl.Program(ctx, """
    // Your code here
    __kernel void SomeKernel(
        __global const float *readOnlyArgument,
        __global float *writeableArgument,
        int someConstant)
    {
        int i = get_global_id(0);
    }
    """).build()
```

{:.code-info cuda}
So how do we allocate memory on the GPU and copy data to and from it. There are
two ways that you can do this. Firstly you can use `cudaMalloc`, where you have
control and choose when to copy to and from the GPU.

{:.code cuda}
```c++
float* numbers = (float*)malloc(n*sizeof(float));
float* numbers_device;
cudaMalloc((void**)&numbers_device, n*sizeof(float));

// Copying to the device
cudaMemcpy(numbers_device, numbers, n*sizeof(float), cudaMemcpyHostToDevice);
// Copying from the device
cudaMemcpy(numbers, numbers_device, n*sizeof(float), cudaMemcpyDeviceToHost);

```


**TODO**: write about running kernels and allocating memory

{:.code-info cuda}
It is also possible to use `cudaMallocManaged`, where copying will be done for
you. This can lead to worse performance.

{:.code cuda}
```c++
float* someMem;
cudaMallocManaged(&someMem, n*sizeof(float));
```
We will be using `cudaMallocManaged` throughout these examples for the sake of
ease and understanding.

When calling our kernel we need to define the dimensions of our kernel and
thread blocks. As said earlier the have up to 1024 threads and each dimension
must be a power of 2. The grid is then defined as the number of thread blocks we
want in each dimension.
```c++
dim3 grid(n,m,1);
dim3 block(16,16,1);

someKernel<<< grid, block >>>(readable, writable, 5.0f);
```

2 Naïve implementation
----------------------
In this version we have taken the naïve approach and done a direct translation
of the program. To use the library for complex arithmetic, we start by writing
`#include "cuComplex.h"`. This enables us to use the type `cuFloatComplex`, and
the functions `cuCmulf` (multiplication of complex numbers) and `cuCaddf`
(addition of complex numbers). Underneath the `cuFloatComplex` type is a vector
type called `float2`, a 2D floating point vector. CUDA has multiple types, which
support vector types, those are char, uchar, short, ushort, int, uint,
long, ulong, longlong, ulonglong, float, and double. The length of the vector
types can be 2, 3, and 4.

The only translation we have done in this version is the `mandelbrot`
function and the complex arithmetic, which means all data is still generated and
sent from the host. But looking at the function we see, that we have to send the
width and height to the function is because we are running in thread blocks, as
described earlier. We could end up out of bounds of our array, which we do not
want and therefore we have this `if`-statement.

{:.code cuda}
```c++
{{#include ../../examples/mandelbrot/cpp/cuda/naive.cu:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/cpp/cuda/naive.ipynb)

{:.code pycuda}
```python
{{#include ../../examples/mandelbrot/python/cuda/naive.py:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/python/cuda/naive.ipynb)

{:.code pyopencl}
```python
{{#include ../../examples/mandelbrot/python/opencl/naive.py:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/python/opencl/naive.ipynb)


3 Less transfer implementation
------------------------------
Transferring data to and from the GPU takes time, which in turn makes our
calculations slower. So we should try to limit how much data we move around.
In the naïve version we generate our data on the host and create a 1000 by 1000
matrix, which is then transfered to the GPU. But we can be smarter than that.
By sending our lists of real and imaginary parts, we can then combine them on
the GPU saving both time and space, because we already have the coordinates of
from our two global ids.

{:.code cuda}
```c++
{{#include ../../examples/mandelbrot/cpp/cuda/lesstransfer.cu:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/cpp/cuda/lesstransfer.ipynb)

{:.code pycuda}
```python
{{#include ../../examples/mandelbrot/python/cuda/lesstransfer.py:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/python/cuda/lesstransfer.ipynb)


{:.code pyopencl}
```python
{{#include ../../examples/mandelbrot/python/opencl/lesstransfer.py:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/python/opencl/lesstransfer.ipynb)

4 GPU only implementation
-------------------------
In this implementation we move the data generation to the GPU. This removes data
transfer to the GPU completely, which will reduce total computation time
considerably, especially when calculating with a higher resolution. Of course we
still need to transfer the result array from the GPU, which is the majority of
our data transfer, but reducing data transfer should be a priority.

{:.code cuda}
```c++
{{#include ../../examples/mandelbrot/cpp/cuda/gpuonly.cu:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/cpp/cuda/gpuonly.ipynb)

{:.code pycuda}
```python
{{#include ../../examples/mandelbrot/python/cuda/gpuonly.py:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/python/cuda/gpuonly.ipynb)

{:.code pyopencl}
```python
{{#include ../../examples/mandelbrot/python/opencl/gpuonly.py:mandelbrot }}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/python/opencl/gpuonly.ipynb)

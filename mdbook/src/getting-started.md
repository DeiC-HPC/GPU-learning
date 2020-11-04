Getting started
===============

In this part we are going to show you how to run a program on a GPU, this is done using an example program, which is converted in a few steps to run on the GPU. As an example program we are going to look at calculating the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set).

0 Before we start
-----------------
Before we start converting the program to run on GPUs, we need to lay some groundwork. We need to understand how to make the program run on the GPU and how to get our data to and from it.

There are a number of different approaches we can choose from.

**CUDA**: CUDA code is a variant of `C++` with some extra syntax. CUDA code typically uses the file-extension `.cu`. The main difference between CUDA code and `C++`, is that you need to declare whether something should be runnable on the CPU, GPU or both. It has special syntax for invoking GPU code from the CPU. To run CUDA code, you can either write entire programs in CUDA and compile them directly using the NVIDIA CUDA Compiler (`nvcc`), or use the PyCUDA library to invoke CUDA code from inside Python.

**OpenMP/OpenACC**: OpenMP and OpenACC are both a collection of pragma-based annotations for `C`, `C++` and `Fortran`. These annotations specify for instance how to parallelize code or when to copy memory. Since these annotations does not affect the syntax of the underlying program, it is mostly possible to compile OpenMP/OpenACC programs while ignoring the annotations ­— however doing so will often change the behavior of the program. OpenMP and OpenACC are very similar and share a lot of history. The main difference between them is that OpenACC was originally designed for GPU parallelization, while OpenMP was designed for CPU parallelization and got GPU support added at a later time.

**OpenCL**: TODO: Write something about OpenCL.

This guide has been written in multiple versions, depending on which platform you want to learn. You can change which version you are viewing to using the drop-down menu at the top. Try changing it now and see how the text below changes:

{:.cuda}
This is text is specific to the CUDA guide.

{:.pycuda}
This is text is specific to the PyCUDA guide.

{:.pyopencl}
This is text is specific to the PyOpenCL guide.

{:.cpp-openmp}
This is text is specific to the OpenMP guide (in C++).

{:.f90-openmp}
This is text is specific to the OpenMP guide (in Fortran).

{:.cpp-openacc}
This is text is specific to the OpenACC guide (in C++).

{:.f90-openacc}
This is text is specific to the OpenACC guide (in Fortran).


0 Sequential implementation
---------------------------
To calculate the Mandelbrot set, we map each point \\(c\\) in the complex plane to a function \\( f_c(z) = z^2 + c \\). The Mandelbrot, is the set of points such that iterated application of \\(f_c\\) remains bounded forever, i.e. \\(|f_c(f_c(\dots f_c(0) \dots))|\\) must not diverge to infinity.

When visualizing the Mandelbrot, one is also interested in how quickly this expression grows beyond the circle bounded by \\(|z|<2\\).

The sequential version contains a function called `mandelbrot`, which is all the logic we need to calculate the Mandelbrot set.

{:.cuda-code cpp-openmp-code cpp-openacc-code}
```c++
{{#include mandelbrot/cpp/reference-implementation.cpp:mandelbrot}}
```

{:.f90-openmp-code f90-openacc-code}
```f90
{{#include mandelbrot/fortran/reference-implementation.f90:mandelbrot}}
```

{:.pycuda-code pyopencl-code}
```python
{{#include mandelbrot/python/reference-implementation.py:mandelbrot}}
```

[Click here to see the entire example](https://github.com). TODO: Add link

It takes a complex number `z` and a maximum number of iterations to be run.

To setup the function we have a lot of variables with default values defining width and height of the image we are generating, how many iterations should at most be run in the `mandelbrot` function, and which area of the fractal should be shown (default is everything).

Then we have two nested loops creating a complex number in the range of the minimum and maximum values and then calculating the mandelbrot function for each of these numbers.

The data is then written to disk so we can visualize it and see the mandelbrot.

{:.cpp-openmp cpp-openacc f90-openmp f90-openacc}
This means that you mostly write your programs as you normally do but with some
exceptions as we need to make it work on the GPU as well.

{:.cpp-openmp cpp-openacc f90-openmp f90-openacc}
When compiling there are several available options, but in this tutorial we will
focus on GCC.

{:.cpp-openacc f90-openacc}
In GCC you add the flag `-fopenacc` when you want the compiler to understand the
OpenACC pragmas. When you compile for the GPU then you will also need to use
`-fno-stack-protector` as these checks will not work and cause the program to
crash. If you use any math libraries then you will also need to add
`-foffload=-lm`.

{:.cpp-openmp f90-openmp}
In GCC you add the flag `-fopenmp` when you want the compiler to understand the
OpenMP pragmas. When you compile for the GPU then you will also need to use
`-fno-stack-protector` as these checks will not work and cause the program to
crash. If you use any math libraries then you will also need to add
`-foffload=-lm`.

{:.cpp-openmp cpp-openacc f90-openmp f90-openacc}
If you are using CUDA 11 then you also have to add `-foffload="-misa=sm_35"` as
the default currently is `sm_30`, which has been dropped. Also the only two
options are `sm_30` and `sm_35`.

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
With the code above we see three variables, blockIdx, blockDim, and threadIdx,
that we have not defined. These will be instantiated when we are running and
tell us know where we are running. When running our code, it is run in a grid of
thread block. Each thread block can have up to 1024 threads and must be a power
of 2.

So how do we allocate memory on the GPU and copy data to and from it. There are
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

```c++
__global__ void mandelbrot(
    const cuFloatComplex *zs,
    int *res,
    ushort width,
    ushort height,
    ushort max_iterations)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    if (x < width && y < height) {
        cuFloatComplex z = zs[x*width+y];
        cuFloatComplex c = z;

        for (int i = 0; i < max_iterations; i++) {
            if (z.x*z.x + z.y*z.y <= 4.0f) {
                res[x*width+y] = i+1;
                z = cuCmulf(z, z);
                z = cuCaddf(z, c);
            }
        }
    }
}
```


3 Less transfer implementation
------------------------------
Transferring data to and from the GPU takes time, which in turn makes our
calculations slower. So we should try to limit how much data we move around.
In the naïve version we generate our data on the host and create a 1000 by 1000
matrix, which is then transfered to the GPU. But we can be smarter than that.
By sending our lists of real and imaginary parts, we can then combine them on
the GPU saving both time and space, because we already have the coordinates of
from our two global ids.

```c++
__global__ void mandelbrot(
    const float *re,
    const float *im,
    int *res,
    ushort width,
    ushort height,
    ushort max_iterations)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    if (x < height && y < width) {
        cuFloatComplex z = make_cuFloatComplex(re[y], im[x]);
        cuFloatComplex c = z;
```

4 GPU only implementation
-------------------------
In this implementation we move the data generation to the GPU. This removes data
transfer to the GPU completely, which will reduce total computation time
considerably, especially when calculating with a higher resolution. Of course we
still need to transfer the result array from the GPU, which is the majority of
our data transfer, but reducing data transfer should be a priority.

```c++
__global__ void mandelbrot(
    int *res,
    ushort width,
    ushort height,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    ushort max_iterations)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    float widthf = width - 1.0f;
    float heightf = height - 1.0f;

    if (x < height && y < width) {
        cuFloatComplex z = make_cuFloatComplex(
            xmin + ((xmax-xmin)*y/widthf),
            ymax - ((ymax-ymin)*x/heightf));
        cuFloatComplex c = z;
```

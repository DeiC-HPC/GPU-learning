# Directives

{:.cuda pycuda opencl pyopencl}
This part of the book is not relevant for you chosen environment. Please go
[here](./native.md) or change environment.

{{#include mandelbrot-sequential-implementation.md}}

1 Before we start
-----------------
Before we start converting the program to run on GPUs, we need to lay some
groundwork. We need to understand how to make the program run on the GPU and how
to get our data to and from it.

{:.code-info cpp-openmp}
OpenMP is a set of pragmas, which the compiler uses to optimize your code. These
directives start with `#pragma omp` where `omp` is the start of the OpenACC
statement.

{:.code-info f90-openmp}
OpenMP is a set of pragmas, which the compiler uses to optimize your code. These
directives start with `$!omp` where `omp` is the start of the OpenACC
statement.

{:.code-info cpp-openacc}
OpenACC is a set of pragmas, which the compiler uses to optimize your code. These
directives start with `#pragma acc` where `acc` is the start of the OpenACC
statement.

{:.code-info f90-openacc}
OpenACC is a set of pragmas, which the compiler uses to optimize your code. These
directives start with `$!acc` where `acc` is the start of the OpenACC
statement.

This means that you mostly write your programs as you normally do but with some
exceptions as we need to make it work on the GPU as well.

When compiling there are several available options, but in this tutorial we will
focus on GCC.

{:.cpp-openacc f90-openacc}
In GCC you add the flag `-fopenacc` when you want the compiler to understand the
OpenACC pragmas.

{:.cpp-openmp f90-openmp}
In GCC you add the flag `-fopenmp` when you want the compiler to understand the
OpenMP pragmas.

When you compile for the GPU then you will also need to use
`-fno-stack-protector` as these checks will not work and cause the program to
crash. If you use any math libraries then you will also need to add
`-foffload=-lm`.

If you are using CUDA 11 then you also have to add `-foffload="-misa=sm_35"` as
the default currently is `sm_30`, which is no longer supported. Also the only two
options are `sm_30` and `sm_35`.

2 Converting to directives
--------------------------
**TODO**: fix this section to be compatible with OpenMP and OpenACC in C++ and
Fortran

The program does not change much from the original when adding the pragmas but
let us take a look.

{:.cpp-openacc}
SOMETHING

{:.cpp-openacc-code}
```c++
{{#include ../../examples/mandelbrot/cpp/openacc/openacc.cpp:mandelbrot}}
```

{:.cpp-openmp}
Around the mandelbrot function there has been added `#pragma omp declare target`
and `#pragma omp end declare target`. this means that it should be compiled for
our offload target.

{:.cpp-openmp-code}
```c++
{{#include ../../examples/mandelbrot/cpp/openmp/openmp.cpp:mandelbrot}}
```

{:.f90-openmp}
Inside the mandelbrot function `$!acc routine` has been added. This means that it
should be compiled for our offload target.

{:.f90-openmp-code}
```f90
{{#include ../../examples/mandelbrot/fortran/openmp/openmp.f90:mandelbrot}}
```

{:.f90-openacc}
Around the mandelbrot function there has been added `$!acc routine`. This means
that it should be compiled for our offload target.

{:.f90-openacc-code}
```f90
{{#include ../../examples/mandelbrot/fortran/openacc/openacc.f90:mandelbrot}}
```

Before the loop we have a pragma
`$!acc parallel loop collapse(2) copyout(res)`
which is where we tell the compiler what we want to happen.
`parallel loop` tells the compiler that the loop is completely parallel and that
every part can be run by itself. The `collapse(2)` tells the compiler that it can
parallelize both loops. At last we have `copyout(res)`, which is a
data clause. Here we say that we want to copy the variable `res` back from the
GPU.  There are also other data clauses besides `copyout`. `copyin` is for
copying data to the GPU. `copy` is for copying data to and from the GPU, so data
is copied to the GPU first and then back when it is done. `create` is for
allocating the variable on the GPU. `present` tells the compiler that it is
already there. There is also `deviceptr` which says that the variable is already
on the GPU and it is containing a pointer to the device memory. This is only
useful when using OpenACC together with another programming model.

**TODO**: Add loops around code as well

{:.cpp-openmp-code}
```c++
{{#include ../../examples/mandelbrot/cpp/openmp/openmp.cpp:loops}}
```
{:.cpp-openacc-code}
```c++
{{#include ../../examples/mandelbrot/cpp/openacc/openacc.cpp:loops}}
```

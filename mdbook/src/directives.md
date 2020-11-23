# Directives

{:.cuda pycuda opencl pyopencl}
This part of the book is not relevant for you chosen environment. Please go
[here](./native.md) or change environment.

{{#include mandelbrot-sequential-implementation.md}}

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

1 Before we start
-----------------
Before we start converting the program to run on GPUs with OpenACC, we need to lay
some groundwork. We need to understand how to make the program run on the GPU and
how to get our data to and from it.

OpenACC is a set of pragmas, which the compiler uses to optimize your code. These
directives start with `$!acc` where `acc` is the start of the OpenACC
statement.

This means that you mostly write your programs as you normally do but with some
exceptions as we need to make it work on the GPU as well.

When compiling there are several available options, but in this tutorial we will
focus on GCC.

In GCC you add the flag `-fopenacc` when you want the compiler to understand the
OpenACC pragmas. When you compile for the GPU then you will also need to use
`-fno-stack-protector` as these checks will not work and cause the program to
crash. If you use any math libraries then you will also need to add
`-foffload=-lm`.

If you are using CUDA 11 then you also have to add `-foffload="-misa=sm_35"` as
the default currently is `sm_30`, which has been dropped. Also the only two
options are `sm_30` and `sm_35`.


2 Converting to OpenACC
----------------------
The program does not change much from the original when turning adding the
OpenACC pragmas but let us take a look.

Around the mandelbrot function there has been added `$!acc routine`. This means
that it should be compiled for our offload target.

Then the main loop has been split into two. That is because you can not write to
disk from the GPU. So we have to copy the data back from the GPU before we can do
that.

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

```f90
!$acc parallel loop collapse(2) copyout(numbers)
do i = 1,n
    do j = 1,n
        numbers(i,j) = mandelbrot( &
            CMPLX( &
                xmin + ((xmax-xmin)*j/(n-1)), &
                ymax - ((ymax-ymin)*i/(n-1)) &
            ), &
            maxi &
        )
    end do
end do
```

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

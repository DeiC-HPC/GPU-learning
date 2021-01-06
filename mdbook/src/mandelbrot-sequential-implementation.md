0 Sequential implementation
---------------------------
To calculate the Mandelbrot set, we map each point \\(c\\) in the complex plane to a function \\( f_c(z) = z^2 + c \\). The Mandelbrot, is the set of points such that iterated application of \\(f_c\\) remains bounded forever, i.e. \\(|f_c(f_c(\dots f_c(0) \dots))|\\) must not diverge to infinity.

When visualizing the Mandelbrot, one is also interested in how quickly this expression grows beyond the circle bounded by \\(|z|<2\\).

The sequential version contains a function called `mandelbrot`, which is all the logic we need to calculate the Mandelbrot set.

{:.code cuda cpp-openmp cpp-openacc}
```c++
{{#include ../../examples/mandelbrot/cpp/reference-implementation.cpp:mandelbrot}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/mandelbrot/cpp/reference-implementation.ipynb)


{:.f90-openmp-code f90-openacc-code}
```f90
{{#include ../../examples/mandelbrot/fortran/reference-implementation.f90:mandelbrot}}
```

{:.pycuda-code pyopencl-code}
```python
{{#include ../../examples/mandelbrot/python/reference-implementation.py:mandelbrot}}
```

It takes a complex number `z` and a maximum number of iterations to be run.

To setup the function we have a lot of variables with default values defining width and height of the image we are generating, how many iterations should at most be run in the `mandelbrot` function, and which area of the fractal should be shown (default is everything).

Then we have two nested loops creating a complex number in the range of the minimum and maximum values and then calculating the mandelbrot function for each of these numbers.

The data is then written to disk so we can visualize it and see the mandelbrot.

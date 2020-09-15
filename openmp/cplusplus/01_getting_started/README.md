Getting started
===============
In this part we are going to show you how to run a program on a GPU, this is
done using an example program, which is converted in a few steps to run on the
GPU. The example program we are going to look at calculates the Mandelbrot set.

0 Sequential implementation
---------------------------
The sequential version contains a function called `mandelbrot`, which is all the
logic we need to calculate the Mandelbrot set.
```c++
int mandelbrot(complex<float> z, int maxiterations) {
    complex<float> c = z;
    for (int i = 0; i < maxiterations; i++) {
        if (abs(z) > 2) {
            return i;
        }
        z = z*z + c;
    }

    return maxiterations;
}
```
It takes a complex number `z` and a maximum number of iterations to be run.

To setup the function we have a lot of variables with default values defining
width and height of the image we are generating, how many iterations should at
most be run in the `mandelbrot` function, and which area of the fractal should
be shown (default is everything).

Then we have two nested loops creating a complex number in the range of the
minimum and maximum values and then calculating the mandelbrot function for each
of these numbers.

The data is then written to disk so we can visualize it and see the mandelbrot.

1 Before we start
-----------------
Before we start converting the program to run on GPUs with OpenMP, we need to lay
some groundwork. We need to understand how to make the program run on the GPU and
how to get our data to and from it.

OpenMP is a set of pragmas, which the compiler uses to optimize your code. These
pragmas start with `#pragma omp` where `omp` is the start of the OpenMP
statement.

This means that you mostly write your programs as you normally do but with some
exceptions as we need to make it work on the GPU as well.

When compiling there are several available options, but in this tutorial we will
focus on GCC.

In GCC you add the flag `-fopenmp` when you want the compiler to understand the
OpenMP pragmas. When you compile for the GPU then you will also need to use
`-fno-stack-protector` as these checks will not work and cause the program to
crash. If you use any math libraries then you will also need to add
`-foffload=-lm`.

If you are using CUDA 11 then you also have to add `-foffload="-misa=sm_35"` as
the default currently is `sm_30`, which has been dropped. Also the only two
options are `sm_30` and `sm_35`.


2 Converting to OpenMP
----------------------
The program does not change much from the original when turning adding the OpenMP
pragmas but let us take a look.

Around the mandelbrot function there has been added `#pragma omp declare target`
and `#pragma omp end declare target`. This means that it should be compiled for
our offload target. We could also have written
`#pragma omp declare target(mandelbrot)`.

Then the main loop has been split into two. That is because you can not write to
disk from the GPU. So we have to copy the data back from the GPU before we can do
that.

Before the loop we have a pragma
`#pragma omp target teams distribute parallel for collapse(2) map(from:res[:height*width])`
which is where we tell the compiler what we want to happen. If we were just
running the code on the CPU would could write
`#pragma omp parallel for collapse(2)`. `parallel for` tells the compiler that
the loop is completely parallel and that every part can be run by itself. The
`collapse(2)` tells the compiler that it can parallelize both loops.
Then we have `target teams distribute`. `target` of course tells that this will
be run on the GPU. `teams` and `distribute` tells the compiler to use more than
thread block on the GPU.
`map` is then about copying data to and from the GPU or allocating data.
We are using the `from`, which tells the compiler that we want to copy
the data from the GPU when it is done using it. It also implicitly tells the
compiler to allocate the array on the GPU. Inside the from statement we have
`res[:height*width]`. Here we say that we want to copy the variable `res` back
from the GPU and the brackets define that we want to copy the range from start to
`height*width`, which is the whole array. The range definition is needed if the
array has been allocated like we did with `malloc`. It can also be used to only
copy parts on an array by defining the start before the colon and the end after.
The other map types that can be used are `to`, `tofrom`, `alloc`, `delete`, and
`release`. `to` copies data to the GPU. `tofrom` copies data to the GPU before
the calculation and then back again afterwards. `alloc` allocates the space on
the GPU. `delete` deletes the data on the GPU. `release` decreases the reference
count (TODO: better explaination).

```c++
#pragma omp target teams distribute parallel for collapse(2) map(from:res[:height*width])
for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
        res[i*width+j] = mandelbrot(
                complex<float>(
                    xmin + ((xmax-xmin)*j/(width-1)),
                    ymax - ((ymax-ymin)*i/(height-1))),
                maxiterations);
    }
}
```

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
Before we start converting the program to run on GPUs with OpenACC, we need to lay
some groundwork. We need to understand how to make the program run on the GPU and
how to get our data to and from it.

OpenACC is a set of pragmas, which the compiler uses to optimize your code. These
pragmas start with `#pragma acc` where `acc` is the start of the OpenACC
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

Around the mandelbrot function there has been added `#pragma acc routine`. This
means that it should be compiled for our offload target.

Then the main loop has been split into two. That is because you can not write to
disk from the GPU. So we have to copy the data back from the GPU before we can do
that.

Before the loop we have a pragma
`#pragma acc parallel loop collapse(2) copyout(res[:width*height])`
which is where we tell the compiler what we want to happen.
`parallel loop` tells the compiler that the loop is completely parallel and that
every part can be run by itself. The `collapse(2)` tells the compiler that it can
parallelize both loops. At last we have `copyout(res[:width*height])`, which is a
data clause. Here we say that we want to copy the variable `res` back from the
GPU and the brackets define that we want to copy the range from start to
`height*width`, which is the whole array. The range definition is needed if the
array has been allocated like we did with `malloc`. It can also be used to only
copy parts on an array by defining the start before the colon and the end after.
There are also other data clauses besides `copyout`. `copyin` is for copying data
to the GPU. `copy` is for copying data to and from the GPU, so data is copied to
the GPU first and then back when it is done. `create` is for allocating the
variable on the GPU. `present` tells the compiler that it is already there. There
is also `deviceptr` which says that the variable is already on the GPU and it is
containing a pointer to the device memory. This is only useful when using OpenACC
together with another programming model.

```c++
#pragma acc parallel loop collapse(2) copyout(res[:width*height])
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

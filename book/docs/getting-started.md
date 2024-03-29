# Getting started

In this part we are going to show you how to run a program on a GPU, this is done using an example program, which is converted in a few steps to run on the GPU. As an example program we are going to look at calculating the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set).

Before we start
---------------
Before we start converting the program to run on GPUs, we need to lay some groundwork. We need to understand how to make the program run on the GPU and how to get our data to and from it.

There are a number of different approaches we can choose from.

**CUDA**: CUDA code is a variant of `C++` with some extra syntax. CUDA code typically uses the file-extension `.cu`. The main difference between CUDA code and `C++`, is that you need to declare whether something should be runnable on the CPU, GPU or both. It has special syntax for invoking GPU code from the CPU. To run CUDA code, you can either write entire programs in CUDA and compile them directly using the NVIDIA CUDA Compiler (`nvcc`), or use the PyCUDA library to invoke CUDA code from inside Python.

**HIP**: HIP code is a near 1:1 in syntax to CUDA and is created to give the same performance, but just compile to both AMD and NVIDIA GPUs. To compile HIP code, you will need the AMD ROCm environment.

**OpenMP/OpenACC**: OpenMP and OpenACC are both a collection of pragma-based annotations for `C`, `C++` and `Fortran`. These annotations specify for instance how to parallelize code or when to copy memory. Since these annotations does not affect the syntax of the underlying program, it is mostly possible to compile OpenMP/OpenACC programs while ignoring the annotations ­— however doing so will often change the behavior of the program. OpenMP and OpenACC are very similar and share a lot of history. The main difference between them is that OpenACC was originally designed for GPU parallelization, while OpenMP was designed for CPU parallelization and got GPU support added at a later time.

This guide has been written with CUDA and HIP in mind. You can change which version you are using with the taps above every specific section. Try it out below.

=== "CUDA"
    This text is specific to the CUDA guide.

=== "HIP"
    This text is specific to the HIP guide.

Choosing the right compiler
---------------------------
Choosing the right environment will most likely be based on what language your
current software project is using. But we will still outline some advantages and
disadvantages of each to help you choose, if you have multiple options.

### CUDA
Advantages
- Native performance

Disadvantages
- Difficult to get optimal performance
- Only works for NVIDIA devices

### HIP
Advantages
- Native performance
- Works with multiple vendors

Disadvantages:
- Difficult to get optimal performance
- Performs worse than CUDA on NVIDIA GPUs in some cases

### OpenMP
Advantages
- Easy to get started with
- Code can be run on both multi core CPU and GPU

Disadvantages
- Performance based on compiler implementation

### OpenACC
Advantages
- Easy to get started with

Disadvantages
- Performance based on compiler implementation

If you think either CUDA or HIP is something for you, then go on with this guide.
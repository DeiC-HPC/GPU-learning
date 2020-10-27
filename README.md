GPU programming
===============

This repo contains material for learning GPU programming in various
programming environments.

Introduction
------------
Before we go ahead and start programming GPUs in any of the programming
languages, I will give you a short introduction to how they work and are
different than general purpose CPUs. This information should help you understand
which problems are suitable to be run on a GPU and something something...

General purpose CPUs are what we call SISD (Single Instruction Stream,
Single Data Stream) which means that every instruction on the CPU is only
working on a single item at a time. This is not entirely true with CPUs
implementing AVX instructions, but let us not get into that here.

A GPU is instead what we call SIMD (Single Instruction, Multiple Data).
This means that everything we do applies to multiple data items at once.

Let us look at an example:
```
i = some_index
a[i] = 42
```

On a CPU we would set a specific index in `a` and to overwrite the whole array
we would need a loop of some sort to overwrite all values.

On a GPU the variable `i` would have a different value for each core, meaning
that we could set all values in `a` with just one line of code instead of using
a loop.

Due to the fact that we are running the same instructions on multiple data
entities at the same time, we will have to think differently about control flow
structures we normally use.

If your code contains an if-else statement both the if and else part will be run
for all data, but only the part which satisfies the condition will be active.
This means that code with big if-else blocks will be ineffective to run on a
GPU.

Compilers used
--------------

- Cuda, NVCC 11
- OpenMP and OpenACC, GCC 10.2.0 with NVPTX offloading.
- Python Cuda, PyCuda 2019.1.2
- Python OpenCL, PyOpenCL 2020.2.2

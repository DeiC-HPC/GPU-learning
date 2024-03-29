# Introduction
This book contains material for learning GPU programming in CUDA and HIP. It is
meant for people who are confident in solving their own problems on a CPU but
needs more computing capabilities.

It is important to note before we start that not all GPUs are equal. So you will
not necessarily see the same performance benefits by implementing your code on
different GPUs. Newer GPUs are also more capable and include many functionalities
that is not available on older devices, for example tensor cores for machine
learning purposes.

Before we begin
---------------
Before we go ahead and start programming GPUs in any of the programming
languages, I will give you a short introduction to how they work and are
different than general purpose CPUs. This information should help you
understand which problems are suitable to be run on a GPU.

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

# Memory Coalescing
In this part we will talk about memory coalescence. We will talk about what it
is and why it is important. We will also showcase a program, where we will see
how it should be done and how it should not be done.

0 What is it?
-------------
On a GPU we have three layers of memory:
- Global
- Shared
- Local (registers)

When we access global memory on a gpu, we access multiple elements at the same
time. This is important to keep in mind, when programming, because access to
global memory is slow. So we need to utilise that we are accessing multiple
elements at the same time. Therefore we need adjacent threads on the GPU to
access adjacent memory in order to gain maximum performance.

1 Matrix addition
-----------------
We will be looking at matrix addition, but for teaching purposes we will only
parallelise one dimension. We will show the differences in parallelising each
dimension and describe why there is a difference.

2 Parallelising the outer loop
------------------------------
When programming a CPU the correct thing to do would be to parallelise the outer
loop, because we would then get cache coherency. So this is what we have done in
the first part. This is not optimal on a GPU, because when we access memory, we
get multiple elements at the same time as described earlier. When parallelising
the outer loop, every thread in the thread block will read their section of
memory, which requires multiple reads of global memory.

![Every thread will read from a different block of memory](notcoalesced.png)

{:.code cuda}
```c++
{{#include ../../examples/matrixaddition/cpp/cuda/naive.cu:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/cpp/cuda/naive.ipynb)

{:.code pycuda}
```c++
{{#include ../../examples/matrixaddition/python/cuda/naive.py:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/python/cuda/naive.ipynb)

{:.code cpp-openmp}
```c++
{{#include ../../examples/matrixaddition/cpp/openmp/naive.cpp:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/cpp/openmp/naive.ipynb)

{:.code cpp-openacc}
```c++
{{#include ../../examples/matrixaddition/cpp/openacc/naive.cpp:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/cpp/openacc/naive.ipynb)

{:.code f90-openmp}
```fortran
{{#include ../../examples/matrixaddition/fortran/openmp/naive.f90:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/fortran/openmp/naive.ipynb)

{:.code f90-openacc}
```fortran
{{#include ../../examples/matrixaddition/fortran/openacc/naive.f90:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/fortran/openacc/naive.ipynb)

3 Parallelising the inner loop
------------------------------
To fix the error in the previous version, we instead parallelise the inner loop.
This means when we are reading data from global memory, then every data point is
given to a thread and no data is fetched without being assigned to a thread.

![All threads read within the same block of memory](coalesced.png)

{:.code cuda}
```c++
{{#include ../../examples/matrixaddition/cpp/cuda/optimized.cu:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/cpp/cuda/optimized.ipynb)

{:.code pycuda}
```c++
{{#include ../../examples/matrixaddition/python/cuda/optimized.py:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/python/cuda/optimized.ipynb)

{:.code cpp-openmp}
```c++
{{#include ../../examples/matrixaddition/cpp/openmp/optimized.cpp:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/cpp/openmp/optimized.ipynb)

{:.code cpp-openacc}
```c++
{{#include ../../examples/matrixaddition/cpp/openacc/optimized.cpp:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/cpp/openacc/optimized.ipynb)

{:.code f90-openmp}
```fortran
{{#include ../../examples/matrixaddition/fortran/openmp/optimized.f90:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/fortran/openmp/optimized.ipynb)

{:.code f90-openacc}
```fortran
{{#include ../../examples/matrixaddition/fortran/openacc/optimized.f90:matrixaddition}}
```
{:.code-link}
[Run the code in Jupyter](/jupyter/lab/tree/matrixaddition/fortran/openacc/optimized.ipynb)

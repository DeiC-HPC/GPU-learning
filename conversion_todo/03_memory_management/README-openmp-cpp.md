Memory management
=================
When using a GPU one of the most important things to do right besides memory
coalescence is memory management. Moving data to and from the GPU takes time,
which could have been used on calculations. Therefore will we in this section
look into how we can better manage our memory transactions.

0 In general
------------
In our previous examples we have moved data between CPU and GPU when doing our
loops. The examples have also been fairly simple with only one loop. When working
with bigger programs with many different loops.

Here we can either use `#pragma omp target data`, where we use curly brackets to
create the scope for the data, or `#pragma omp target enter data` and
`#pragma omp target exit data`.

1 

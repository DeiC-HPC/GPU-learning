# Memory management

When using a GPU one of the most important things to do right besides memory
coalescence is memory management. Moving data to and from the GPU takes time,
which could have been used on calculations. Therefore will we in this section
look into how we can better manage our memory transactions.

0 In general
------------
In our previous examples we have moved data between CPU and GPU when doing our
loops. The examples have also been fairly simple with only one loop. When working
with bigger programs with many different loops.

{:.cpp-openmp}
Here we can either use `#pragma omp target data`, where we use curly brackets to
create the scope for the data, or `#pragma omp target enter data` and
`#pragma omp target exit data`.

{:.cpp-openacc}
Here we can either use `#pragma acc data`, where we use curly brackets to
create the scope for the data, or `#pragma acc enter data` and
`#pragma acc exit data`.

{:.f90-openmp}
Here we can either use `$!omp target data`, where we use curly brackets to
create the scope for the data, or `$!omp target enter data` and
`$!omp target exit data`.

{:.f90-openacc}
Here we can either use `$!acc data`, where we use curly brackets to create the
scope for the data, or `$!acc enter data` and `$!acc exit data`.

1 Map reduce example
--------------------
To show how much a difference it makes and how to use data transfer, we will now
look at a map reduce example. Map reduce refers to two operations normally used
in functional programming. A map is where you do the same operation, for example
adding two to each element, over every element in a list. Reduce is then taking a
list and then reducing it to a single element. This could for example be getting
the sum a list. If you're doing a reduction loop in OpenACC then you have to add
the `reduction` clause to your OpenACC pragma. Inside the the reduction clause
you set a reduction operator and then a number of variables. The possible
reduction operators are `+`, `*`, `max`, `min`, `&`, `|`, `^`, `&&`, and `||`.

The two programs are based around two loops the first being a map and the second
being a reduce. In the not optimized program we copy variables in both loops.
```f90
!$acc parallel loop copyout(elements)
do i=1,num
    elements(i) = i
enddo

!$acc parallel loop copyin(elements) reduction(+:res)
do i=1,num
    res = res + elements(i)
enddo
```
In the optimized version we put the code into a data region and create the
`elements` array on the GPU and then do our calculations so it is never copied.
The only variable that is copied is the `res`. As copying variables and arrays
between CPU and GPU is an expensive operation then the goal is to limit that.
```f90
!$acc data create(elements)
!$acc parallel loop
do i=1,num
    elements(i) = i
enddo

!$acc parallel loop reduction(+:res)
do i=1,num
    res = res + elements(i)
enddo
!$acc end data
```
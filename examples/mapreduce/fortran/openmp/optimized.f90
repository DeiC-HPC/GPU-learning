PROGRAM mapreduceoptimized
    integer, parameter :: num = 100000000
    integer*8 :: res = 0
    integer, dimension(:), allocatable :: elements
    real :: start_time, end_time

    allocate(elements(num))

    call cpu_time(start_time)
    ! ANCHOR: mapreduce
    !$omp target data map(alloc: elements)
    !$omp target teams distribute parallel do
    do i=1,num
        elements(i) = i
    enddo

    !$omp target teams distribute parallel do reduction(+: res) map(from: res)
    do i=1,num
        res = res + elements(i)
    enddo
    !$omp end target data
    ! ANCHOR_END: mapreduce
    call cpu_time(end_time)

    print *, "time:", end_time - start_time, "seconds"
    print *, "the result is:", res
END PROGRAM mapreduceoptimized

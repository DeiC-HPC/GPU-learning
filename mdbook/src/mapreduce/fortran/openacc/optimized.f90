PROGRAM mapreduceoptimized
    integer, parameter :: num = 100000000
    integer*8 :: res = 0
    integer, dimension(:), allocatable :: elements
    real :: start_time, end_time

    allocate(elements(num))

    call cpu_time(start_time)
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
    call cpu_time(end_time)

    print *, "time:", end_time - start_time, "seconds"
    print *, "the result is:", res
END PROGRAM mapreduceoptimized

PROGRAM matrixaddition
    integer, parameter :: height = 10000
    integer, parameter :: width  = 10000
    integer, dimension(:,:), allocatable :: a
    integer, dimension(:,:), allocatable :: b
    integer, dimension(:,:), allocatable :: res
    logical :: allElementsAre2 = .true.
    real :: start_time, end_time

    allocate(a(width,height))
    allocate(b(width,height))
    allocate(res(width,height))
    a(:,:) = 1
    b(:,:) = 1

    call cpu_time(start_time)
    /* ANCHOR: matrixaddition */
    !$acc parallel loop copyin(a) copyin(b) copyout(res)
    do i = 1,height
        do j = 1,width
            res(i,j) = a(i,j) + b(i,j)
        enddo
    enddo
    /* ANCHOR_END: matrixaddition */
    call cpu_time(end_time)

    do i = 1,height
        do j = 1,width
            if (res(i,j) .ne. 2) then
                allElementsAre2 = .false.
            endif
        enddo
    enddo

    print *, "time:", end_time - start_time, "seconds"
    if (allElementsAre2) then
        print *, "All numbers in matrix are 2"
    else
        print *, "Not all numbers in matrix are 2"
    endif

END PROGRAM matrixaddition

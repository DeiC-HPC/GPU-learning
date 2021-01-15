! ANCHOR: mandelbrot
function mandelbrot(z, maxiterations) result(iterations)
    complex :: z, c
    integer maxinterations, iterations
    !$acc routine

    c = z
    do iterations = 0, maxiterations-1
        if (abs(z) > 2) then
            return
        endif
        z = z*z + c
    end do
end function mandelbrot
! ANCHOR_END: mandelbrot

PROGRAM mandelbrot_openacc
    integer, parameter :: n = 1000
    integer, parameter :: maxi = 100
    integer, parameter :: out_unit = 20
    real :: start, finish
    integer, dimension(:,:), allocatable :: numbers
    real :: ymin, ymax, xmin, xmax
    real :: start_time, end_time
    ymin = -2.0
    ymax = 2.0
    xmin = -2.5
    xmax = 1.5

    allocate(numbers(n,n))

    call cpu_time(start_time)
    ! ANCHOR: loops
    !$acc parallel loop collapse(2) copyout(numbers)
    do i = 1,n
        do j = 1,n
            numbers(i,j) = mandelbrot( &
                CMPLX( &
                    xmin + ((xmax-xmin)*j/(n-1)), &
                    ymin + ((ymax-ymin)*i/(n-1)) &
                ), &
                maxi &
            )
        end do
    end do
    ! ANCHOR_END: loops
    call cpu_time(end_time)

    open(out_unit, file = 'mandelbrot_openacc.csv')
    do i = 1,n
        do j = 1,n
            if (j .eq. n) then
                write(out_unit,*) numbers(i,j)
            else
                write(out_unit,"(I3,A)",advance="no") numbers(i,j), ","
            endif
        enddo
    enddo
    close(out_unit)
    deallocate(numbers)

    print *, "time:", end_time - start_time, "seconds"

END PROGRAM mandelbrot_openacc

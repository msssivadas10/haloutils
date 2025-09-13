module integrate_mod
    !! Helper module for numerical integration.

    use iso_c_binding
    implicit none

    private
    public :: leggauss, int_heap_push, int_heap_pop

    integer, parameter :: dp = c_double

    real(c_double), parameter :: PI = 3.141592653589793_c_double
    !! Pi

    real(c_double), parameter, public :: K15(2, 8) = reshape([ &
    !   nodes               , weights               
        0.000000000000000_dp, 0.209482141084728_dp, & !*
        0.207784955007898_dp, 0.204432940075298_dp, &
        0.405845151377397_dp, 0.190350578064785_dp, & !*
        0.586087235467691_dp, 0.169004726639267_dp, &
        0.741531185599394_dp, 0.140653259715525_dp, & !*
        0.864864423359769_dp, 0.104790010322250_dp, &
        0.949107912342759_dp, 0.063092092629979_dp, & !*
        0.991455371120813_dp, 0.022935322010529_dp  &
    ], shape(K15))
    !! Nodes and weigths for Kronrod-15 rule. Exact for polynomials upto 
    !! degree 29. Points are symmetric about x=0.
    
    real(c_double), parameter, public :: G7(2, 4) = reshape([ &
    !   nodes               , weights               
        0.000000000000000_dp, 0.417959183673469_dp, & !*
        0.405845151377397_dp, 0.381830050505119_dp, & !*
        0.741531185599394_dp, 0.279705391489277_dp, & !*
        0.949107912342759_dp, 0.129484966168870_dp  & !*
    ], shape(G7))
    !! Nodes and weights of for Gauss-7 rule. Exact for polynomials upto 
    !! degree 13. Points are symmetric about x=0 and subset of Kronrod-15
    !! rule.
    
contains

    subroutine leggauss(n, x, w) 
        !! Generate Gauss-Legendre quadrature rule of order N for [-1, 1].

        integer(c_int), intent(in), value  :: n
        !! Order of the rule: number of nodes.

        real(c_double), intent(out) :: x(n)
        !! Nodes 

        real(c_double), intent(out) :: w(n)
        !! Weights

        real(c_double) :: xj, xjo, pm, pn, ptmp
        integer(c_int) :: j, k

        ! If order is odd number, x = 0 is a node
        if ( modulo(n, 2) == 1 ) then
            ! Calculating Legendre polynomial P_n(0) using its reccurence relation
            xj = 0._c_double
            pm = 0._c_double
            pn = 1._c_double
            do k = 0, n-1
                ptmp = -k*pm / (k + 1._c_double)
                pm   = pn
                pn   = ptmp
            end do
            x(n/2 + 1) = 0._c_double
            w(n/2 + 1) = 2._c_double / (n*pm)**2 ! weight 
        end if

        ! Other nodes (roots of the n-th Legendre polynomial)
        do j = 1, n/2
            xj  = cos( (2*j - 0.5_c_double) * PI / (2*n + 1._c_double) ) ! Initial guess for the root
            xjo = 100._c_double 
            do while ( abs(xj - xjo) > 1e-08_c_double )
                ! Calculating Legendre polynomial P_n(xj) using its reccurence relation
                pm = 0._c_double
                pn = 1._c_double
                do k = 0, n-1
                    ptmp = ( (2*k + 1)*xj*pn - k*pm ) / (k + 1._c_double)
                    pm   = pn
                    pn   = ptmp
                end do
                xjo = xj
                xj  = xj - pn * (xj**2 - 1) / (n*xj*pn - n*pm)
            end do
            x(j)     = -xj
            w(j)     =  2*(1 - xj**2) / (n*xj*pn - n*pm)**2 !! weight
            x(n-j+1) =  xj
            w(n-j+1) =  w(j)
        end do
    
    end subroutine leggauss

    subroutine int_heap_push(heap, size, capacity, a, b, val, err)
        !! Push the integration result on a interval [a, b] to the interval heap.

        real(c_double), intent(inout) :: heap(4, capacity) 
        !! Heap array
        
        integer(c_int64_t), intent(inout) :: size
        !! Current size of the heap

        integer(c_int64_t), intent(in) :: capacity
        !! Maximum capacity of the heap

        real(c_double), intent(in) :: a
        !! Left end value of the interval

        real(c_double), intent(in) :: b
        !! Right end value of the interval

        real(c_double), intent(in) :: val
        !! Value of the integral in the interval

        real(c_double), intent(in) :: err
        !! Error estimate

        integer(c_int64_t) :: i, p
        real(c_double)     :: temp(4)

        if ( size >= capacity ) stop 'error: heap full'
        size       = size + 1
        i          = size
        heap(:, i) = [ -err, a, b, val ]

        ! Bubble up
        do while (i > 1)
            p = i / 2
            if (heap(1,i) < heap(1,p)) then
                temp(:)   = heap(:,i)
                heap(:,i) = heap(:,p)
                heap(:,p) = temp(:)
                i = p
            else
                exit
            end if
        end do
        
    end subroutine int_heap_push

    subroutine int_heap_pop(heap, size, capacity, a, b, val, err)
        !! Pop the interval with largest error from the heap.

        real(c_double), intent(inout) :: heap(4, capacity) 
        !! Heap array
        
        integer(c_int64_t), intent(inout) :: size
        !! Current size of the heap

        integer(c_int64_t), intent(in) :: capacity
        !! Maximum capacity of the heap

        real(c_double), intent(out) :: a
        !! Left end value of the interval

        real(c_double), intent(out) :: b
        !! Right end value of the interval

        real(c_double), intent(out) :: val
        !! Value of the integral in the interval

        real(c_double), intent(out) :: err
        !! Error estimate

        integer(c_int64_t) :: i, c
        real(c_double)     :: temp(4)

        if (size <= 0) stop "error: heap empty"
        
        ! Return min element
        err = -heap(1,1) 
        a   =  heap(2,1)
        b   =  heap(3,1)
        val =  heap(4,1)
        
        ! Move last to root
        heap(:,1) = heap(:,size) 
        size      = size - 1

        ! Bubble down
        i = 1
        do while (2*i <= size)
            c = 2*i
            if (c < size .and. heap(1,c+1) < heap(1,c)) c = c + 1
            if (heap(1,c) < heap(1,i)) then
                temp(:)   = heap(:,i)
                heap(:,i) = heap(:,c)
                heap(:,c) = temp(:)
                i = c
            else
                exit
            end if
        end do
        
    end subroutine int_heap_pop

end module integrate_mod

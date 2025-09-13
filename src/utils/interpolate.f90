module interpolate_mod
    !! Natural cubic spline interpolation.

    use iso_c_binding
    implicit none

    private
    public :: generate_cspline, interpolate
    
contains

    subroutine generate_cspline(n, spline) bind(c)
        !! Generate a natural cubic spline for the points (x, y)

        integer(c_int64_t), intent(in), value :: n
        !! Number of points

        real(c_double), intent(inout) :: spline(3,n)
        !! Spline parameters. Its first two columns give the X and Y
        !! values for interpolation.

        real(c_double) :: a(n), b(n), c(n), d(n), w
        integer(c_int64_t) :: i

        ! Creating the tridiagonal system to solve
        a(1)     = 0._c_double
        a(2:n)   = 1._c_double / ( spline(1,2:n) - spline(1,1:n-1) )     
        b(1)     = 2._c_double * a(2)
        b(2:n-1) = 2._c_double * ( a(2:n-1) + a(3:n) )
        b(n)     = 2._c_double * a(n)   
        c(1:n-1) = a(2:n)
        c(n)     = 0._c_double
        d(1)     = 3._c_double * ( spline(2,2) - spline(2,1  ) ) * a(2)
        d(2:n-1) = 3._c_double * ( &
                    ( spline(2,2:n-1) - spline(2,1:n-2) ) * a(2:n-1) + &
                    ( spline(2,3:n  ) - spline(2,2:n-1) ) * a(3:n  )   &
                )
        d(n)     = 3._c_double * ( spline(2,n) - spline(2,n-1) ) * a(n)

        ! Forward elimination with partial pivoting
        do i = 1, n-1
            ! Partial pivoting: swap row i and i+1 if needed
            if (abs(b(i)) < abs(a(i+1))) then
                ! Swap b(i) <-> b(i+1)
                w = b(i)
                b(i)   = b(i+1) 
                b(i+1) = w
                ! Adjust sub-diagonal
                w = a(i+1) 
                a(i+1) = a(i) 
                a(i)   = 0._c_double
                ! Swap c(i) <-> c(i+1) if valid
                if (i < n-1) then
                    w = c(i) 
                    c(i)   = c(i+1) 
                    c(i+1) = w
                end if
                ! Swap d(i) <-> d(i+1)
                w = d(i)
                d(i)   = d(i+1)
                d(i+1) = w
            end if

            ! Elimination
            w = a(i+1) / b(i)
            b(i+1) = b(i+1) - w * c(i)
            d(i+1) = d(i+1) - w * d(i)
            a(i+1) = 0._c_double
        end do

        ! Back substitution
        spline(3,n) = d(n) / b(n)
        do i = n-1, 1, -1
            spline(3,i) = (d(i) - c(i)*spline(3,i+1)) / b(i)
        end do

    end subroutine generate_cspline

    function interpolate(x, n, spline) result(res) bind(c)
        !! Calculate the interpolated value using a natural cubic spline.

        real(c_double), intent(in), value :: x
        !! Value at which data interpolated
        
        integer(c_int64_t), intent(in), value :: n
        !! Number of points
        
        real(c_double), intent(in) :: spline(3,n)
        !! Spline parameters. Its first two columns give the X and Y
        !! values for interpolation.

        real(c_double) :: res, dx, dy, u, v, a, b
        integer(c_int64_t) :: i1, i2, i

        i1 = 1
        i2 = n
        do while (i2-i1 > 1)
            i = (i2 + i1) / 2
            if ( spline(1,i) > x ) then
                i2 = i
            else
                i1 = i
            end if
        end do

        dx  = spline(1,i2) - spline(1,i1)
        dy  = spline(2,i2) - spline(2,i1)
        u   = (x - spline(1,i1)) / dx
        v   = 1._c_double - u
        a   =  spline(3,i1) * dx - dy
        b   = -spline(3,i2) * dx + dy
        res = v*spline(2,i1) + u*spline(2,i2) + u*v*( v*a + u*b )
        
    end function interpolate
    
end module interpolate_mod
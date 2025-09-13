module rfunctions_mod
    !! Calculation of matter variance and correlation from power spectrum data.
    
    use, intrinsic :: ieee_arithmetic
    use iso_c_binding
    implicit none

    private
    public :: variance, correlation

    real(c_double), parameter :: pi  = 3.141592653589793_c_double
    !! Pi
    
contains

! Matter Variance:

    function variance(lnr, j, nu, filt, pktab, size, cls) result(res) bind(C)
        !! Calculate the value of smoothed matter variance for j=0 and nu=0.
        !!
        !! Calculate the j-th order spectral moment of the density field, smoothed 
        !! using the specified filter function at a radius of r. 

        real(c_double), intent(in), value :: lnr
        !! Natural log of smoothing radius in Mpc

        integer(c_int), intent(in), value :: j
        !! Order of the moment (j=0 will give matter variance)

        integer(c_int), intent(in), value :: nu
        !! If non-zero, calculate first derivative w.r.to r

        integer(c_int), intent(in), value :: filt
        !! Code of filter function (0=tophat, 1=gaussian)
        
        real(c_double), intent(in) :: pktab(cls, size)
        !! Precalculated power spectrum table. The columns should be
        !! 1=Nodes of integration (natural log of k in 1/Mpc), 
        !! 2=Value of natural log of power spectrum, 
        !! 3=Weights for integration.
        
        integer(c_int64_t), intent(in), value :: size
        !! Size of the power spectrum table.

        integer(c_int), intent(in), value :: cls
        !! Columns of the power spectrum table: must be 2 or 3. If 3, use the 
        !! the last column as weights. Otherwise, Simpson's rule is used. For 
        !! other values, return value will be NaN.

        real(c_double) :: res
        !! Calculated spectral moment value  

        integer(c_int) :: args(3)

        args(1) = j        
        args(2) = nu      
        args(3) = filt  
        res     = integrate(lnr, variance_integrand, args, pktab, size, cls)

    end function variance

    function variance_integrand(lnr, lnk, lnp, args) result(res)
        !! Calculate the value of integrand for variance integral. For j>0, 
        !! Return the j-th moment of the matter power spectrum.

        real(c_double), intent(in) :: lnr
        !! Natural log of smoothing radius in Mpc
        
        real(c_double), intent(in) :: lnk
        !! Natural log of wavenumber in 1/Mpc

        real(c_double), intent(in) :: lnp
        !! Natural log of power spectrum value

        integer(c_int), intent(in) :: args(3)
        !! Arguments

        integer(c_int) :: j, nu, filt
        real(c_double) :: res, kr, w0, w1
        
        j    = args(1) ! Order of the moment (j=0 will give matter variance)
        nu   = args(2) ! If non-zero, calculate first derivative w.r.to r
        filt = args(3) ! Code of filter function (0=tophat, 1=gaussian)

        ! Dimenssionless matter power spectrum
        res = exp( lnk * (3 + 2*j) + lnp ) 
        
        ! Filter function calculation
        kr  = exp( lnk + lnr )
        select case ( filt )
        case ( 1 )
            ! Gaussian filter
            w0 = exp( -0.5_c_double * kr**2 )
            w1 = w0
            if ( nu == 1 ) then
                ! First derivative
                w1 = -kr * w0
            end if 
        case default
            ! Spherical tophat filter
            w0 = 3*( sin(kr) - kr * cos(kr) ) / kr**3
            w1 = w0
            if ( nu == 1 ) then
                ! First derivative
                w1 = ( 3*(kr**2 - 3) * sin(kr) + 9*kr * cos(kr) ) / kr**4
            end if 
        end select
        if ( nu == 1 ) w1 = 2*w1 * exp(lnk)
        
        res = res * w0 * w1 

    end function variance_integrand

! Matter Correlation:

    function correlation(lnr, j, pktab, size, cls) result(res) bind(C)
        !! Calculate the value of matter correlation function.

        real(c_double), intent(in), value :: lnr
        !! Natural log of smoothing radius in Mpc

        integer(c_int), intent(in), value :: j
        !! Non-zero value for j will give average matter correlation
        
        real(c_double), intent(in) :: pktab(cls, size)
        !! Precalculated power spectrum table. The nodes (lnk, first column) 
        !! should correspond to that of 7-th order Gauss-Legendre quadrature. 
        !! Main interval can be subdivided further and the nodes in each sub-
        !! interval can be stacked together, for more accuracy. 

        integer(c_int64_t), intent(in), value :: size
        !! Size of the power spectrum table.

        integer(c_int), intent(in), value :: cls
        !! Columns of the power spectrum table: must be 2 or 3. If 3, use the 
        !! the last column as weights. Otherwise, Simpson's rule is used. For 
        !! other values, return value will be NaN.

        real(c_double) :: res
        !! Calculated spectral moment value  

        integer(c_int) :: args(3)

        args(1) = j        
        res     = integrate(lnr, correlation_integrand, args, pktab, size, cls)
        
    end function correlation

    function correlation_integrand(lnr, lnk, lnp, args) result(res)
        !! Calculate the value of integrand for variance integral.

        real(c_double), intent(in) :: lnr
        !! Natural log of smoothing radius in Mpc
        
        real(c_double), intent(in) :: lnk
        !! Natural log of wavenumber in 1/Mpc

        real(c_double), intent(in) :: lnp
        !! Natural log of power spectrum value

        integer(c_int), intent(in) :: args(3)
        !! Arguments

        integer(c_int) :: j
        real(c_double) :: res, kr
        
        j = args(1) ! Non-zero value for j will give average matter correlation

        ! Dimenssionless matter power spectrum
        res = exp( lnk * 3 + lnp ) 
        
        ! Filter function calculation
        kr  = exp( lnk + lnr )
        if ( j == 0 ) then 
            res = res * ( sin(kr) / kr )
        else
            ! For average correlation function: this integral converges faster
            res = res * ( 3*( sin(kr) - kr * cos(kr) ) / kr**3 )
        end if

    end function correlation_integrand

! Helper functions:

    function integrate(lnr, f, args, pktab, size, cls) result(res)
        !! Calculate the value of integral.

        real(c_double), intent(in), value :: lnr
        !! Natural log of smoothing radius in Mpc

        interface
            function f(lnr_, lnk_, lnp_, args_) result(y)
                !! Function to integrate
                import :: c_double, c_int
                real(c_double), intent(in) :: lnr_, lnk_, lnp_
                integer(c_int), intent(in) :: args_(3) 
                real(c_double) :: y
            end function
        end interface

        integer(c_int), intent(in) :: args(3)
        !! Arguments
        
        real(c_double), intent(in) :: pktab(cls, size)
        !! Precalculated power spectrum table

        integer(c_int64_t), intent(in), value :: size
        !! Size of the power spectrum table

        integer(c_int), intent(in), value :: cls
        !! Columns of the power spectrum table

        real(c_double) :: res
        !! Calculated integral value  

        integer(c_int64_t) :: i, m
        real(c_double)     :: lnk, fkr, s, delta(2), weight(3)

        res = 0.0_c_double
        if ( cls == 2 ) then
            !! Use Simpson's rule integration
            
            do i = 1, size-2, 2
                ! For odd number of intervals, this loops runs up to the second
                ! last interval... 

                ! Calculating weights:
                delta = pktab(1, i+1:i+2) - pktab(1, i:i+1)
                if( abs( delta(2) - delta(1) ) < 1.0e-08_c_double ) then
                    ! Weights for uniform Simpson's rule
                    weight = [ 1.0_c_double, 4.0_c_double, 1.0_c_double ]
                else
                    ! Weights for non-uniform Simpson's rule
                    weight = [ &
                        2.0_c_double - delta(2) / delta(1), &
                        sum(delta)**2 / product(delta),     &
                        2.0_c_double - delta(1) / delta(2)  &
                    ]
                end if

                s = 0.0_c_double
                do m = i, i+2
                    lnk = pktab(1, m)
                    fkr = pktab(2, m)
                    fkr = f(lnr, lnk, fkr, args) 
                    s   = s + fkr * weight(m-i+1)
                end do
                res = res + sum(delta) * s
            
            end do 
            ! For odd number of intervals, the last two intervals are handled 
            ! seperately (ref:  <https://en.wikipedia.org/wiki/Simpson%27s_rule>)
            if ( modulo(size-1, 2) == 1 ) then
                i = size-2

                ! Calculating weights:
                delta = pktab(1, i+1:i+2) - pktab(1, i:i+1)
                if( abs( delta(2) - delta(1) ) < 1.0e-08_c_double ) then
                    ! Weights for uniform Simpson's rule
                    weight = [ 1.0_c_double, 4.0_c_double, 1.0_c_double ]
                else
                    ! Weights for non-uniform Simpson's rule
                    s      = sum(delta)
                    weight = [ &
                        delta(2) * ( 2*delta(2) + 3*delta(1) ) / sum(delta), &
                        delta(2) * (   delta(2) + 3*delta(1) ) / delta(1),   &
                        delta(2)**3 /  delta(1) / sum(delta)                 &
                    ]
                end if

                s = 0.0_c_double
                do m = i, i+2
                    lnk = pktab(1, m)
                    fkr = pktab(2, m)
                    fkr = f(lnr, lnk, fkr, args) 
                    s   = s + fkr * weight(m-i+1)
                end do
                res = res + sum(delta) * s

            end if
            res = res / 6.0_c_double

        else if ( cls == 3 ) then
            !! Use integration with col=3 as the weights
            do i = 1, size
                lnk = pktab(1, i)
                fkr = pktab(2, i)
                fkr = f(lnr, lnk, fkr, args) 
                res = res + fkr * pktab(3, i)
            end do

        else
            !! Invalid value: result will be NaN
            res = ieee_value(res, ieee_quiet_nan)
            return
        end if
        res = res / ( 2*pi**2 )
        
    end function integrate

end module rfunctions_mod

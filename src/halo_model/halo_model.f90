module halo_model_mod
    !! A module for basic halo model calculations. These calculations are based  
    !! on a 5-parameter halo occupation distribution (HOD) model. 

    use iso_fortran_env, only: stderr => error_unit
    use iso_c_binding
    use constants_mod
    use random_mod
    use interpolate_mod
    use integrate_mod
    implicit none

    private
    public :: lagrangian_r, central_count, satellite_count, subhalo_mass_function,  &
              halo_concentration, average_halo_density, average_galaxy_density,     &
              average_satellite_frac, average_galaxy_bias, optimize_model

    type, public, bind(c) :: hmargs_t
        !! A struct storing various halomodel parameters
        real(c_double) :: lnm_min    !! Minimum halo mass (in Msun) to have at least one central galaxy 
        real(c_double) :: sigma_m    !! Width of the central galaxy transition range. (0 for a step function)
        real(c_double) :: lnm0       !! Minimum halo mass (in Msun) to have satellite galaxies
        real(c_double) :: lnm1       !! Scale factor for power law satellite count relation (Msun)
        real(c_double) :: alpha      !! Index for the  power law satellite count relation
        real(c_double) :: scale_shmf !! Scale parameter for the subhalo mass-function
        real(c_double) :: slope_shmf !! Slope parameter for the subhalo mass-function
        real(c_double) :: z          !! Redshift
        real(c_double) :: H0         !! Hubble parameter value
        real(c_double) :: Om0        !! Total matter density parameter
        real(c_double) :: Delta_m    !! Matter overdensity w.r.to mean background density
        real(c_double) :: dplus      !! Growth factor at this redshift
    end type

contains

    function lagrangian_r(args, lnm) result(lnr) bind(c)
        !! Return the Lagrangian radius a halo (natural log of value in Mpc), 
        !! given its mass. 

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: lnm
        !! Natural log of halo mass (Msun)

        real(c_double) :: lnr, rho_m, rho_h

        rho_m = args%Om0 * ( critical_density_const * args%H0**2 ) ! Matter density at z=0 in Msun/Mpc^3 
        rho_h = rho_m ! Halo density (TODO: chek if the halo density is rho_m * self.Delta)

        ! Lagrangian radius (r) corresponding to halo mass
        lnr = ( lnm + log(3._c_double / (4*pi) / rho_h ) ) / 3._c_double ! r in Mpc
        
    end function lagrangian_r

    function central_count(args, lnm) result(res) bind(c)
        !! Return the average count of central galaxies in a halo, given its 
        !! mass. This will be a sigmoid function with smoothness controlled by 
        !! the `sigma_m` parameter. If it is 0, then it will be a step function.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: lnm
        !! Natural log of halo mass (Msun)

        real(c_double) :: res
        res = central_count_(lnm, args%lnm_min, args%sigma_m)
        
    end function central_count

    function central_count_(lnm, lnm_min, sigma_m) result(res)
        real(c_double), intent(in) :: lnm, lnm_min, sigma_m
        real(c_double) :: res
    
        res = lnm - lnm_min
        if ( abs(sigma_m) < 1e-06 ) then
            ! Heaviside step function
            if ( res < 0._c_double ) then 
                res = 0._c_double
            else
                res = 1._c_double
            end if
        else
            ! Sigmoid function
            res = 0.5_c_double*( 1._c_double + erf(res / sigma_m) )
        end if

    end function central_count_

    function satellite_count(args, lnm) result(res) bind(c)
        !! Return the average count of satellite galaxies in a halo, given 
        !! its mass. Average fraction of satellites is given by a power law 
        !! of the form `((m0 - m)/ m1)^alpha`.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: lnm
        !! Natural log of halo mass (Msun)

        real(c_double) :: res        
        res = satellite_frac_(lnm, args%lnm0, args%lnm1, args%alpha) 
        if ( res > 0._c_double ) res = res * central_count(args, lnm)

    end function satellite_count

    function satellite_frac_(lnm, lnm0, lnm1, alpha) result(res)
        real(c_double), intent(in) :: lnm, lnm0, lnm1, alpha
        real(c_double) :: res

        res = ( exp(lnm) - exp(lnm0) ) / exp(lnm1)
        if ( res < 0._c_double ) then
            res = 0._c_double
        else
            res = res**alpha 
        end if 
        
    end function satellite_frac_

    function subhalo_mass_function(args, x, lnm) result(res) bind(c)
        !! Calculate the subhalo mass-function for given halo mass. This 
        !! is a bounded power law defined in the subhalo masses in the 
        !! range `[m_min, scale*m]`.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: x
        !! Mass of the subhalo as fraction of the parent halo

        real(c_double), intent(in), value :: lnm
        !! Natural log of halo mass (Msun)

        real(c_double) :: res, a, b, c, p
        
        a = exp(args%lnm_min - lnm)
        b = args%scale_shmf
        if ( x < a .or. x > b ) then
            res = 0._c_double
            return
        end if
        
        p   = args%slope_shmf ! power law index
        c   = p * a**p / (1._c_double - (a / b)**p) ! amplitude
        res = c * x**(-p - 1)
    
    end function subhalo_mass_function

    function halo_concentration(args, sigma) result(res) bind(c)
        !! Return the value of halo concentration parameter for a given 
        !! mass, calculated for the current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: sigma
        !! Matter variance corresponding to the halo mass

        real(c_double) :: res, zp1, c0, b, g1, g2, v0, v, t

        ! Redshift dependent parameters:
        zp1 = (1._c_double + args%z)
        c0  = 3.395_c_double * zp1**(-0.215_c_double )
        b   = 0.307_c_double * zp1**( 0.540_c_double )
        g1  = 0.628_c_double * zp1**(-0.047_c_double )
        g2  = 0.317_c_double * zp1**(-0.893_c_double )
        v0  = ( &
                4.135_c_double                   &
                    - 0.564_c_double   * zp1     &
                    - 0.210_c_double   * zp1**2  &
                    + 0.0557_c_double  * zp1**3  &
                    - 0.00348_c_double * zp1**4  &
             ) / args%dplus
        v  = delta_sc / sigma
        t  = v / v0

        ! Concentration-mass relation:
        res = c0 * t**(-g1) * (1._c_double + t**(1._c_double / b))**(-b*(g2 - g1))
        
    end function halo_concentration

! Halo averages:

    subroutine halo_average1(flg, lnma, lnmb, hod, mfns, mfspline,   &
                             abstol, reltol, maxiter, res, err, stat &
        )
        !! Calculate the halo average without bias weight in the interval [lnma, lnmb].
        character(len=1)  , intent(in)  :: flg !! Weight selector (c=central count, s=satellite count, n=no weight)
        real(c_double)    , intent(in)  :: lnma
        real(c_double)    , intent(in)  :: lnmb
        real(c_double)    , intent(in)  :: hod(5) !! Model parameter values
        real(c_double)    , intent(in)  :: mfspline(3,mfns) !! Mass-function spline
        integer(c_int64_t), intent(in)  :: mfns
        real(c_double)    , intent(in)  :: abstol  !! Absolute tolerance
        real(c_double)    , intent(in)  :: reltol  !! Relative tolerance
        integer(c_int64_t), intent(in)  :: maxiter !! Maximum number of iterations
        real(c_double)    , intent(out) :: res
        real(c_double)    , intent(out) :: err
        integer(c_int)    , intent(out) :: stat !! Error code: 0=ok, 1=integral not converged

        integer(c_int64_t) :: iter
        real(c_double)     :: xa, xb, xm, I0, I1, I2, err0, err1, err2
        integer(c_int64_t) :: heap_size, heap_capacity
        real(c_double), allocatable :: heap(:, :)

        heap_size     = 0
        heap_capacity = 10*maxiter ! Heap capacity
        allocate( heap(4, heap_capacity) )

        ! Initial evaluation
        call halo_average1_(flg, lnma, lnmb, hod, mfns, mfspline, res, err)
        call int_heap_push(heap, heap_size, heap_capacity, lnma, lnmb, I0, err0)

        res  = I0
        err  = err0
        stat = 1
        do iter = 1, maxiter

            ! Stop if tolerance is met
            if ( err <= max(abstol, reltol*abs(res)) ) then
                stat = 0 
                exit
            endif

            ! Pop worst interval
            call int_heap_pop(heap, heap_size, heap_capacity, xa, xb, I0, err0)

            xm = 0.5_c_double * (xa + xb)
            
            ! Refine on left interval
            call halo_average1_(flg, xa, xm, hod, mfns, mfspline, I1, err1)
            call int_heap_push(heap, heap_size, heap_capacity, xa, xm, I1, err1) ! Push new interval back
            
            ! Refine on left interval
            call halo_average1_(flg, xm, xb, hod, mfns, mfspline, I2, err2)
            call int_heap_push(heap, heap_size, heap_capacity, xm, xb, I2, err2) ! Push new interval back
            
            ! Update global sums
            res = res + (I1   + I2   - I0  ) ! replace old interval
            err = err + (err1 + err2 - err0)
            
        end do

        deallocate(heap)    
        
    end subroutine halo_average1

    ! PRIVATE:
    subroutine halo_average1_(flg, lnma, lnmb, hod, mfns, mfspline, res, err)
        !! Calculate the halo average with bias weight in the interval [lnma, lnmb]. 
        character(len=1)  , intent(in)  :: flg
        real(c_double)    , intent(in)  :: lnma
        real(c_double)    , intent(in)  :: lnmb
        real(c_double)    , intent(in)  :: hod(5) ! HOD parameters
        integer(c_int64_t), intent(in)  :: mfns
        real(c_double)    , intent(in)  :: mfspline(3,mfns)
        real(c_double)    , intent(out) :: res
        real(c_double)    , intent(out) :: err

        integer(c_int64_t) :: j
        real(c_double)     :: intg, intk, xj, fj, fj_, scale

        scale = 0.5_c_double * (lnmb - lnma)

        xj = lnma + scale
        fj = exp( interpolate(xj, mfns, mfspline) )
        select case ( flg )
        case ( 'c' ) 
            ! Using central galaxy count as weight
            fj = fj * central_count_(xj, hod(1), hod(2))
        case ( 's' ) 
            ! Using satellite galaxy count as weight
            fj = fj * central_count_(xj, hod(1), hod(2))
            if ( abs(fj) > 0. ) &
                fj = fj * satellite_frac_(xj, hod(3), hod(4), hod(5))
        ! Default: no weight
        end select 
        intk  = fj * K15(2,1) 
        intg  = fj *  G7(2,1) 
        do j = 2, 8

            xj = lnma + scale * (1. - K15(1,j))  
            fj = exp( interpolate(xj, mfns, mfspline) ) 
            select case ( flg )
            case ( 'c' ) 
                ! Using central galaxy count as weight
                fj = fj * central_count_(xj, hod(1), hod(2))
            case ( 's' ) 
                ! Using satellite galaxy count as weight
                fj = fj * central_count_(xj, hod(1), hod(2))
                if ( abs(fj) > 0. ) &
                    fj = fj * satellite_frac_(xj, hod(3), hod(4), hod(5))
            ! Default: no weight
            end select 
            fj_ = fj 

            xj = lnma + scale * (1. + K15(1,j)) 
            fj = exp( interpolate(xj, mfns, mfspline) ) 
            select case ( flg )
            case ( 'c' ) 
                ! Using central galaxy count as weight
                fj = fj * central_count_(xj, hod(1), hod(2))
            case ( 's' ) 
                ! Using satellite galaxy count as weight
                fj = fj * central_count_(xj, hod(1), hod(2))
                if ( abs(fj) > 0. ) &
                    fj = fj * satellite_frac_(xj, hod(3), hod(4), hod(5))
            ! Default: no weight
            end select 
            fj = fj + fj_
            
            intk = intk + fj * K15(2,j)
            if ( mod(j, 2) == 1 ) intg = intg + fj * G7(2,(j+1)/2) ! Point also in G7 rule
        end do
        
        intk = scale * intk
        intg = scale * intg
        res  = intk
        err  = abs(intk - intg)
        
    end subroutine halo_average1_

    subroutine halo_average2(flg, lnma, lnmb, hod, mfns, mfspline, bfns,       &
                             bfspline, abstol, reltol, maxiter, res, err, stat &
        )
        !! Calculate the halo average without bias weight in the interval [lnma, lnmb].
        character(len=1)  , intent(in)  :: flg !! Weight selector (c=central count, s=satellite count, n=no weight)
        real(c_double)    , intent(in)  :: lnma
        real(c_double)    , intent(in)  :: lnmb
        real(c_double)    , intent(in)  :: hod(5) !! Model parameter values
        integer(c_int64_t), intent(in)  :: mfns
        real(c_double)    , intent(in)  :: mfspline(3,mfns) !! Mass-function spline
        integer(c_int64_t), intent(in)  :: bfns
        real(c_double)    , intent(in)  :: bfspline(3,bfns) !! Bias spline
        real(c_double)    , intent(in)  :: abstol  !! Absolute tolerance
        real(c_double)    , intent(in)  :: reltol  !! Relative tolerance
        integer(c_int64_t), intent(in)  :: maxiter !! Maximum number of iterations
        real(c_double)    , intent(out) :: res
        real(c_double)    , intent(out) :: err
        integer(c_int)    , intent(out) :: stat !! Error code: 0=ok, 1=integral not converged

        integer(c_int64_t) :: iter
        real(c_double)     :: xa, xb, xm, I0, I1, I2, err0, err1, err2
        integer(c_int64_t) :: heap_size, heap_capacity
        real(c_double), allocatable :: heap(:, :)

        heap_size     = 0
        heap_capacity = 10*maxiter ! Heap capacity
        allocate( heap(4, heap_capacity) )

        ! Initial evaluation
        call halo_average2_(flg, lnma, lnmb, hod, mfns, mfspline, &
                            bfns, bfspline, res, err              &
        )
        call int_heap_push(heap, heap_size, heap_capacity, lnma, lnmb, I0, err0)

        res  = I0
        err  = err0
        stat = 1
        do iter = 1, maxiter

            ! Stop if tolerance is met
            if ( err <= max(abstol, reltol*abs(res)) ) then
                stat = 0 
                exit
            endif

            ! Pop worst interval
            call int_heap_pop(heap, heap_size, heap_capacity, xa, xb, I0, err0)

            xm = 0.5_c_double * (xa + xb)
            
            ! Refine on left interval
            call halo_average2_(flg, xa, xm, hod, mfns, mfspline, &
                                bfns, bfspline, I1, err1          &
            )
            call int_heap_push(heap, heap_size, heap_capacity, xa, xm, I1, err1) ! Push new interval back
            
            ! Refine on left interval
            call halo_average2_(flg, xm, xb, hod, mfns, mfspline, &
                                bfns, bfspline, I2, err2          &
            )
            call int_heap_push(heap, heap_size, heap_capacity, xm, xb, I2, err2) ! Push new interval back
            
            ! Update global sums
            res = res + (I1   + I2   - I0  ) ! replace old interval
            err = err + (err1 + err2 - err0)
            
        end do

        deallocate(heap)    
        
    end subroutine halo_average2

    ! PRIVATE:
    subroutine halo_average2_(flg, lnma, lnmb, hod, mfns, mfspline, &
                              bfns, bfspline, res, err               &
        )
        !! Calculate the halo average with bias weight in the  interval [lnma, lnmb]. 
        character(len=1)  , intent(in)  :: flg
        real(c_double)    , intent(in)  :: lnma
        real(c_double)    , intent(in)  :: lnmb
        real(c_double)    , intent(in)  :: hod(5)
        integer(c_int64_t), intent(in)  :: mfns
        real(c_double)    , intent(in)  :: mfspline(3,mfns)
        integer(c_int64_t), intent(in)  :: bfns
        real(c_double)    , intent(in)  :: bfspline(3,bfns)
        real(c_double)    , intent(out) :: res
        real(c_double)    , intent(out) :: err

        integer(c_int64_t) :: j
        real(c_double)     :: intg, intk, xj, fj, fj_, scale

        scale = 0.5_c_double * (lnmb - lnma)

        xj = lnma + scale ! log(m)
        fj = exp( interpolate(xj, mfns, mfspline) ) 
        fj = fj * exp( interpolate(xj, bfns, bfspline) )
        select case ( flg )
        case ( 'c' ) 
            ! Using central galaxy count as weight
            fj = fj * central_count_(xj, hod(1), hod(2))
        case ( 's' ) 
            ! Using satellite galaxy count as weight
            fj = fj * central_count_(xj, hod(1), hod(2))
            if ( abs(fj) > 0. ) &
                fj = fj * satellite_frac_(xj, hod(3), hod(4), hod(5))
        ! Default: no weight
        end select  
        intk  = fj * K15(2,1) 
        intg  = fj *  G7(2,1) 
        do j = 2, 8

            xj = lnma + scale * (1. - K15(1,j))  
            fj = exp( interpolate(xj, mfns, mfspline) ) 
            fj = fj * exp( interpolate(xj, bfns, bfspline) )
            select case ( flg )
            case ( 'c' ) 
                ! Using central galaxy count as weight
                fj = fj * central_count_(xj, hod(1), hod(2))
            case ( 's' ) 
                ! Using satellite galaxy count as weight
                fj = fj * central_count_(xj, hod(1), hod(2))
                if ( abs(fj) > 0. ) &
                    fj = fj * satellite_frac_(xj, hod(3), hod(4), hod(5))
            ! Default: no weight
            end select 
            fj_ = fj 

            xj = lnma + scale * (1. + K15(1,j)) 
            fj = exp( interpolate(xj, mfns, mfspline) ) 
            fj = fj * exp( interpolate(xj, bfns, bfspline) ) 
            select case ( flg )
            case ( 'c' ) 
                ! Using central galaxy count as weight
                fj = fj * central_count_(xj, hod(1), hod(2))
            case ( 's' ) 
                ! Using satellite galaxy count as weight
                fj = fj * central_count_(xj, hod(1), hod(2))
                if ( abs(fj) > 0. ) &
                    fj = fj * satellite_frac_(xj, hod(3), hod(4), hod(5))
            ! Default: no weight
            end select 
            fj = fj + fj_
            
            intk = intk + fj * K15(2,j)
            if ( mod(j, 2) == 1 ) intg = intg + fj * G7(2,(j+1)/2) ! Point also in G7 rule
        end do
        
        intk = scale * intk
        intg = scale * intg
        res  = intk
        err  = abs(intk - intg)
        
    end subroutine halo_average2_

    function average_halo_density(args, lnma, lnmb, mfns, mfspline, &
                                  abstol, reltol, maxiter           &
        ) result(res) bind(c)
        !! Return the average halo number density for the given halo mass 
        !! range at current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: lnma
        !! Lower limit: natural log of halo mass (Msun)
        
        real(c_double), intent(in), value :: lnmb
        !! Upper limit: natural log of halo mass (Msun)

        real(c_double), intent(in) :: mfspline(3,mfns)
        !! Cubic spline for halo mass function, log(dn/dlnm) as function 
        !! of log(halo mass)

        integer(c_int64_t), intent(in), value :: mfns
        !! Size of the mass-function spline data

        real(c_double), intent(in), value :: abstol
        !! Absolute tolerance

        real(c_double), intent(in), value :: reltol
        !! Relative tolerance

        integer(c_int64_t), intent(in), value :: maxiter
        !! Maximum number of iterations for calculating integral

        real(c_double) :: res, err, hod(5)
        integer(c_int) :: stat

        ! HOD parameters
        hod = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ] 
        
        call halo_average1('n', lnma, lnmb, hod, mfns, mfspline,   &
                           abstol, reltol, maxiter, res, err, stat &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_halo_density: integral failed to converge'
        
    end function average_halo_density

    function average_galaxy_density(args, lnma, lnmb, mfns, mfspline, &
                                    abstol, reltol, maxiter           &
        ) result(res) bind(c)
        !! Return the average galaxy number density for the given halo mass 
        !! range at current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: lnma
        !! Lower limit: natural log of halo mass (Msun)
        
        real(c_double), intent(in), value :: lnmb
        !! Upper limit: natural log of halo mass (Msun)

        real(c_double), intent(in) :: mfspline(3,mfns)
        !! Cubic spline for halo mass function, log(dn/dlnm) as function 
        !! of log(halo mass)

        integer(c_int64_t), intent(in), value :: mfns
        !! Size of the mass-function spline data

        real(c_double), intent(in), value :: abstol
        !! Absolute tolerance

        real(c_double), intent(in), value :: reltol
        !! Relative tolerance

        integer(c_int64_t), intent(in), value :: maxiter
        !! Maximum number of iterations for calculating integral

        real(c_double) :: res, res1, res2, err, hod(5)
        integer(c_int) :: stat

        ! HOD parameters
        hod = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ] 
        
        ! Average central galaxy density
        call halo_average1('c', lnma, lnmb, hod, mfns, mfspline,    &
                           abstol, reltol, maxiter, res1, err, stat &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_galaxy_density: integral (c) failed to converge'
        
        ! Average satellite galaxy density
        call halo_average1('s', lnma, lnmb, hod, mfns, mfspline,    &
                           abstol, reltol, maxiter, res2, err, stat &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_galaxy_density: integral (s) failed to converge'

        ! Average galaxy density
        res = res1 + res2

    end function average_galaxy_density

    function average_satellite_frac(args, lnma, lnmb, mfns, mfspline, &
                                    abstol, reltol, maxiter           &
        ) result(res) bind(c)
        !! Return the average satellite fraction for the given halo mass 
        !! range at current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: lnma
        !! Lower limit: natural log of halo mass (Msun)
        
        real(c_double), intent(in), value :: lnmb
        !! Upper limit: natural log of halo mass (Msun)

        real(c_double), intent(in) :: mfspline(3,mfns)
        !! Cubic spline for halo mass function, log(dn/dlnm) as function 
        !! of log(halo mass)

        integer(c_int64_t), intent(in), value :: mfns
        !! Size of the mass-function spline data

        real(c_double), intent(in), value :: abstol
        !! Absolute tolerance

        real(c_double), intent(in), value :: reltol
        !! Relative tolerance

        integer(c_int64_t), intent(in), value :: maxiter
        !! Maximum number of iterations for calculating integral

        real(c_double) :: res, res1, res2, err, hod(5) 
        integer(c_int) :: stat

        ! HOD parameters
        hod = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ] 
        
        ! Average central galaxy density
        call halo_average1('c', lnma, lnmb, hod, mfns, mfspline,    &
                           abstol, reltol, maxiter, res1, err, stat &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_satellite_frac: integral (c) failed to converge'
        
        ! Average satellite galaxy density
        call halo_average1('s', lnma, lnmb, hod, mfns, mfspline,    &
                           abstol, reltol, maxiter, res2, err, stat &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_satellite_frac: integral (s) failed to converge'

        ! Average satellite fraction
        res = res2 / (res1 + res2)
        
    end function average_satellite_frac

    function average_galaxy_bias(args, lnma, lnmb, mfns, mfspline, bfns, &
                                 bfspline, abstol, reltol, maxiter       &
        ) result(res) bind(c)
        !! Return the average satellite fraction for the given halo mass 
        !! range at current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: lnma
        !! Lower limit: natural log of halo mass (Msun)

        real(c_double), intent(in), value :: lnmb
        !! Upper limit: natural log of halo mass (Msun)

        real(c_double), intent(in) :: mfspline(3,mfns)
        !! Cubic spline for halo mass function, log(dn/dlnm) as function 
        !! of log(halo mass)

        integer(c_int64_t), intent(in), value :: mfns
        !! Size of the mass-function spline data

        real(c_double), intent(in) :: bfspline(3,bfns)
        !! Bias spline

        integer(c_int64_t), intent(in), value :: bfns
        !! Size of the bias spline

        real(c_double), intent(in), value :: abstol
        !! Absolute tolerance

        real(c_double), intent(in), value :: reltol
        !! Relative tolerance

        integer(c_int64_t), intent(in), value :: maxiter
        !! Maximum number of iterations for calculating integral

        real(c_double) :: res, res1, res2, res3, res4, err, hod(5)
        integer(c_int) :: stat

        ! HOD parameters
        hod = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ] 

        ! Average central galaxy density
        call halo_average1('c', lnma, lnmb, hod, mfns, mfspline,    &
                           abstol, reltol, maxiter, res1, err, stat &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_galaxy_bias: integral (c) failed to converge'

        ! Average satellite galaxy density
        call halo_average1('s', lnma, lnmb, hod, mfns, mfspline,    &
                           abstol, reltol, maxiter, res2, err, stat &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_galaxy_bias: integral (s) failed to converge'

        ! Average central galaxy bias
        call halo_average2('c', lnma, lnmb, hod, mfns, mfspline, bfns, bfspline,   &
                           abstol, reltol, maxiter, res3, err, stat                &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_galaxy_bias: integral (bc) failed to converge'

        ! Average satellite galaxy bias
        call halo_average2('c', lnma, lnmb, hod, mfns, mfspline, bfns, bfspline,   &
                           abstol, reltol, maxiter, res4, err, stat                &
        )
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: average_galaxy_bias: integral (bs) failed to converge'

        ! Average galaxy bias
        res = (res3 + res4) / (res1 + res2)

    end function average_galaxy_bias

! Optimizing HOD model

    subroutine optimize_model(args, ngal_target, fsat_target, bounds, flag, &
                              lnma, lnmb, mfns, mfspline, abstol, reltol,   &
                              maxiter, maxsteps, stat                       &
        ) bind(c)
        !! Calculate the optimum HOD parameters to make the calculated values of
        !! galaxy density and satellite fraction match their obsereved values. 
        !! (NOTE: NOT TESTED)
        
        type(hmargs_t), intent(inout) :: args
        !! Model parameter values

        real(c_double), intent(in), value :: ngal_target
        !! Observed value of galaxy density

        real(c_double), intent(in), value :: fsat_target
        !! Observed value of satellite fraction

        real(c_double), intent(in) :: bounds(5,2)
        !! Lower and upper bounds for the values. To fix the value of a 
        !! parameter, set lower bound equal to upper bound. 

        integer(c_int), intent(in), value :: flag
        !! If non-zero, then set the value of `lnm0` to `lnm_min`.  

        real(c_double), intent(in), value :: lnma
        !! Lower limit for integration: natural log of halo mass (Msun)

        real(c_double), intent(in), value :: lnmb
        !! Upper limit for integration: natural log of halo mass (Msun)

        real(c_double), intent(in) :: mfspline(3,mfns)
        !! Cubic spline for halo mass function, log(dn/dlnm) as function 
        !! of log(halo mass)

        integer(c_int64_t), intent(in), value :: mfns
        !! Size of the mass-function spline data

        real(c_double), intent(in), value :: abstol
        !! Absolute tolerance value for checking convergence

        real(c_double), intent(in), value :: reltol
        !! Relative tolerance value for checking convergence 

        integer(c_int64_t), intent(in), value :: maxiter
        !! Maximum number of iterations

        integer(c_int64_t), intent(in), value :: maxsteps
        !! Maximum number of steps for optimzation

        integer(c_int), intent(out) :: stat
        !! Status (0=success, 1=not converged)

        integer :: i, fixed_param(5)
        integer(c_int64_t) :: k
        real(c_double) :: lower_bound(5), upper_bound(5), x(5), grad(5), xt(5), &
                          xa, xb, cost, step, trial_cost, g2
        real(c_double), parameter :: c1   = 1.e-04_c_double
        real(c_double), parameter :: beta = 0.5_c_double

        lower_bound = max( bounds(1:5, 1), [ -6.d0, 0.d0  , -6.d0, -6.d0, -10.d0 ] )
        upper_bound = min( bounds(1:5, 2), [ 20.d0, 1.d+04, 20.d0, 20.d0,  10.d0 ] )
        fixed_param = merge( &
            [1, 1, 1, 1, 1], &
            [0, 0, 0, 0, 0], &
            abs(lower_bound - upper_bound) < 1e-08_c_double &
        )

        x = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ]
        x = min( max(lower_bound, x), upper_bound ) ! clipping between bounds
        if ( flag /= 0 ) x(3) = x(1) ! set m_min and m0 parameters the same

        ! Evaluate initial cost
        cost = model_cost_(x, ngal_target, fsat_target, lnma, lnmb, &
                           mfns, mfspline, abstol, reltol, maxiter  &
        )

        stat = 1
        do k = 1, maxsteps

            x = min( max(lower_bound, x), upper_bound ) ! clipping between bounds
            
            ! Calculating the gradients numerically
            do i = 1, 5
                if ( fixed_param(i) /= 0 ) then
                    ! This parameter is fixed, so the gradient is set to zero, so 
                    ! that no variation is applied along that direction.
                    grad(i) = 0._c_double
                else if ( flag /= 0 .and. i == 3 ) then 
                    ! Since m0=m_min is forced, use the calculated gradient for 
                    ! parameter m_min for m0:
                    grad(3) = grad(1)
                else
                    xt(1:5) = x(1:5)
                    step    = max( 1e-04_c_double, 1e-06_c_double*abs( x(i) ) )
                    ! For left point
                    xa      = max( lower_bound(i), x(i) - step )
                    xt(i)   = xa
                    if ( flag /= 0 ) xt(3) = xt(1) 
                    grad(i) = model_cost_(xt, ngal_target, fsat_target, lnma, lnmb, &
                                          mfns, mfspline, abstol, reltol, maxiter   &
                    )
                    ! For right point
                    xb      = min( upper_bound(i), x(i) + step )
                    xt(i)   = xb
                    if ( flag /= 0 ) xt(3) = xt(1) 
                    grad(i) = model_cost_(xt, ngal_target, fsat_target, lnma, lnmb, &
                                          mfns, mfspline, abstol, reltol, maxiter   &
                    ) - grad(i)
                    ! Derivative approx. in this direction
                    grad(i) = grad(i) / max(xb - xa, 1e-12_c_double)
                end if
            end do

            ! Stop if tolerance is met
            g2 = dot_product(grad, grad)
            if ( g2 <= abstol**2 ) then
                stat = 0 
                exit
            endif

            ! Line search along the gradient direction
            step = 1._c_double
            do 
               xt = x - step * grad 
               xt = min( max(lower_bound, xt), upper_bound ) ! clipping between bounds
               if ( flag /= 0 ) xt(3) = xt(1) 

               trial_cost = model_cost_(xt, ngal_target, fsat_target, lnma, lnmb, &
                                        mfns, mfspline, abstol, reltol, maxiter   &
                )
                if ( trial_cost < cost - c1*step*g2 ) then
                    exit
                else
                    step = beta*step
                    if ( step < 1e-12_c_double ) exit 
                end if
            end do

            ! Accepting the trial values
            x    = xt
            cost = trial_cost

        end do

    end subroutine optimize_model

    function model_cost_(hod, ngal_target, fsat_target, lnma, lnmb, mfns, &
                         mfspline, abstol, reltol, maxiter                &
        ) result(cost)
        !! Cost function for HOD optimization 
        real(c_double)    , intent(in)        :: hod(5)
        real(c_double)    , intent(in), value :: ngal_target
        real(c_double)    , intent(in), value :: fsat_target
        real(c_double)    , intent(in), value :: lnma
        real(c_double)    , intent(in), value :: lnmb
        real(c_double)    , intent(in)        :: mfspline(3,mfns)
        integer(c_int64_t), intent(in), value :: mfns
        real(c_double)    , intent(in), value :: abstol
        real(c_double)    , intent(in), value :: reltol
        integer(c_int64_t), intent(in), value :: maxiter
        
        real(c_double) :: cost, ngal, fsat, res1, res2, err
        integer(c_int) :: stat
        
        ! Average central galaxy density
        call halo_average1('c', lnma, lnmb, hod, mfns, mfspline,    &
                           abstol, reltol, maxiter, res1, err, stat &
        ) 
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: optimize_model: integral (c) failed to converge'

        ! Average satellite galaxy density
        call halo_average1('s', lnma, lnmb, hod, mfns, mfspline,    &
                           abstol, reltol, maxiter, res2, err, stat &
        ) 
        if ( stat /= 0 ) &
            write(stderr,'(a)') 'warning: optimize_model: integral (s) failed to converge'

        ngal = res1 + res2 ! Average total galaxy density
        fsat = res2 / ngal ! Average satellite fraction
        cost = (ngal / ngal_target - 1)**2 + (fsat / fsat_target - 1)**2 ! Cost

    end function model_cost_

end module halo_model_mod

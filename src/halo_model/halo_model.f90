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
              average_satellite_frac, average_galaxy_bias, halo_average 

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

    subroutine halo_average(flg, use_weight, use_bias, hod, cols, tabsize, hftab, res)
        !! Calculate the halo average in the interval [lnma, lnmb], with optional 
        !! halo bias weight. 

        character(1)      , intent(in)  :: flg
        integer(c_int)    , intent(in)  :: cols, use_bias, use_weight
        integer(c_int64_t), intent(in)  :: tabsize
        real(c_double)    , intent(in)  :: hod(5), hftab(cols,tabsize)
        real(c_double)    , intent(out) :: res

        integer(c_int64_t) :: i, m
        real(c_double)     :: lnm, fm, nm, s, delta(2), weight(3)

        res = 0.0_c_double
        if ( use_weight /= 0 ) then
            ! Use integration with last column (4 if using bias weighting, otherwise 3) 
            ! as weights for an integral rule . 
            write(stderr,'("info: halo_average: ", a)') 'using weights'

            do i = 1, tabsize
                lnm = hftab(1, i)

                ! Calculating integrand:
                if ( use_bias == 0 ) then
                    fm = exp(hftab(2, i)) ! dn/dlnm
                else
                    fm = exp(hftab(2, i) + hftab(3, i)) ! dn/dlnm * bias
                end if
                select case ( flg )
                case ( 'c' )
                    ! Central galaxy count only
                    nm = central_count_( lnm, hod(1), hod(2) )
                    fm = nm * fm
                case ( 's' )
                    ! Satellite galaxy count only
                    nm = central_count_( lnm, hod(1), hod(2) ) ! central galaxy count 
                    nm = nm * satellite_frac_( lnm, hod(3), hod(4), hod(5) ) ! satellite count
                    fm = nm * fm
                case ( 't' )
                    ! Total (central + satellite) galaxy count
                    nm = central_count_( lnm, hod(1), hod(2) ) ! central galaxy count 
                    nm = nm + nm * satellite_frac_( lnm, hod(3), hod(4), hod(5) ) ! total count
                    fm = nm * fm
                end select

                res = res + fm * hftab(cols, i)
            end do
        
        else
            !! Use Simpson's rule integration
            write(stderr,'("info: halo_average: ", a)') 'using simpson rule'
            
            do i = 1, tabsize-2, 2
                ! For odd number of intervals, this loops runs up to the second
                ! last interval... 

                ! Calculating weights:
                delta = hftab(1, i+1:i+2) - hftab(1, i:i+1)
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
                    lnm = hftab(1, m)

                    ! Calculating integrand:
                    if ( use_bias == 0 ) then
                        fm = exp(hftab(2, m)) ! dn/dlnm
                    else
                        fm = exp(hftab(2, m) + hftab(3, m)) ! dn/dlnm * bias
                    end if
                    select case ( flg )
                    case ( 'c' )
                        ! Central galaxy count only
                        nm = central_count_( lnm, hod(1), hod(2) )
                        fm = nm * fm
                    case ( 's' )
                        ! Satellite galaxy count only
                        nm = central_count_( lnm, hod(1), hod(2) ) ! central galaxy count 
                        nm = nm * satellite_frac_( lnm, hod(3), hod(4), hod(5) ) ! satellite count
                        fm = nm * fm
                    case ( 't' )
                        ! Total (central + satellite) galaxy count
                        nm = central_count_( lnm, hod(1), hod(2) ) ! central galaxy count 
                        nm = nm + nm * satellite_frac_( lnm, hod(3), hod(4), hod(5) ) ! total count
                        fm = nm * fm
                    end select

                    s = s + fm * weight(m-i+1)
                end do
                res = res + sum(delta) * s
            
            end do 
            ! For odd number of intervals, the last two intervals are handled 
            ! seperately (ref:  <https://en.wikipedia.org/wiki/Simpson%27s_rule>)
            if ( modulo(tabsize-1, 2) == 1 ) then
                i = tabsize-2

                ! Calculating weights:
                delta = hftab(1, i+1:i+2) - hftab(1, i:i+1)
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
                    lnm = hftab(1, m)

                    ! Calculating integrand:
                    if ( use_bias == 0 ) then
                        fm = exp(hftab(2, m)) ! dn/dlnm
                    else
                        fm = exp(hftab(2, m) + hftab(3, m)) ! dn/dlnm * bias
                    end if
                    select case ( flg )
                    case ( 'c' )
                        ! Central galaxy count only
                        nm = central_count_( lnm, hod(1), hod(2) )
                        fm = nm * fm
                    case ( 's' )
                        ! Satellite galaxy count only
                        nm = central_count_( lnm, hod(1), hod(2) ) ! central galaxy count 
                        nm = nm * satellite_frac_( lnm, hod(3), hod(4), hod(5) ) ! satellite count
                        fm = nm * fm
                    case ( 't' )
                        ! Total (central + satellite) galaxy count
                        nm = central_count_( lnm, hod(1), hod(2) ) ! central galaxy count 
                        nm = nm + nm * satellite_frac_( lnm, hod(3), hod(4), hod(5) ) ! total count
                        fm = nm * fm
                    end select

                    s = s + fm * weight(m-i+1)
                end do
                res = res + sum(delta) * s

            end if
            res = res / 6.0_c_double

        end if
        
    end subroutine halo_average

    function average_halo_density(args, cols, tabsize, hftable) result(res) bind(c)
        !! Return the average halo number density for the given halo mass 
        !! range at current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in) :: hftable(cols,tabsize)
        !! Table of halo mass function, log(dn/dlnm) as function of log(halo mass)

        integer(c_int), intent(in), value :: cols
        !! Number of columns in mass-function table. It should be 2 or 3, if 
        !! integration weights are specified in the 3-rd column.

        integer(c_int64_t), intent(in), value :: tabsize
        !! Size of the mass-function table.

        real(c_double) :: res, hod(5)
        integer(c_int) :: use_weight

        hod = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ] ! HOD parameters

        use_weight = 0_c_int
        if ( cols < 2 ) then
            write (stderr, '("error: average_halo_density: ",a)') 'insufficient data'
            res = -1._c_double
            return
        else if ( cols > 2 ) then
            use_weight = 1_c_int ! use last column as weights
        end if

        call halo_average('n', use_weight, 0_c_int, hod, cols, tabsize, hftable, res)
        
    end function average_halo_density

    function average_galaxy_density(args, cols, tabsize, hftable) result(res) bind(c)
        !! Return the average galaxy number density for the given halo mass 
        !! range at current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in) :: hftable(cols,tabsize)
        !! Table of halo mass function, log(dn/dlnm) as function of log(halo mass)

        integer(c_int), intent(in), value :: cols
        !! Number of columns in mass-function table. It should be 2 or 3, if 
        !! integration weights are specified in the 3-rd column.

        integer(c_int64_t), intent(in), value :: tabsize
        !! Size of the mass-function table.

        real(c_double) :: res, hod(5)
        integer(c_int) :: use_weight

        hod = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ] ! HOD parameters

        use_weight = 0_c_int
        if ( cols < 2 ) then
            write (stderr, '("error: average_galaxy_density: ",a)') 'insufficient data'
            res = -1._c_double
            return
        else if ( cols > 2 ) then
            use_weight = 1_c_int ! use last column as weights
        end if

        call halo_average('t', use_weight, 0_c_int, hod, cols, tabsize, hftable, res)
        
    end function average_galaxy_density

    function average_satellite_frac(args, cols, tabsize, hftable) result(res) bind(c)
        !! Return the average satellite fraction for the given halo mass 
        !! range at current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in) :: hftable(cols,tabsize)
        !! Table halo mass function, log(dn/dlnm) as function of log(halo mass)

        integer(c_int), intent(in), value :: cols
        !! Number of columns in mass-function table. It should be 2 or 3, if 
        !! integration weights are specified in the 3-rd column.

        integer(c_int64_t), intent(in), value :: tabsize
        !! Size of the mass-function table.

        real(c_double) :: res, res1, res2, hod(5)
        integer(c_int) :: use_weight

        hod = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ] ! HOD parameters

        use_weight = 0_c_int
        if ( cols < 2 ) then
            write (stderr, '("error: average_satellite_frac: ",a)') 'insufficient data'
            res = -1._c_double
            return
        else if ( cols > 2 ) then
            use_weight = 1_c_int ! use last column as weights
        end if

        ! Average central galaxy density:
        call halo_average('c', use_weight, 0_c_int, hod, cols, tabsize, hftable, res1)
        
        ! Average satellite galaxy density:
        call halo_average('s', use_weight, 0_c_int, hod, cols, tabsize, hftable, res2)
        
        ! Average satellite fraction:
        res = res2 / (res1 + res2)
        
    end function average_satellite_frac

    function average_galaxy_bias(args, cols, tabsize, hftable) result(res) bind(c)
        !! Return the average satellite fraction for the given halo mass 
        !! range at current redshift.

        type(hmargs_t), intent(in) :: args
        !! Model parameter values

        real(c_double), intent(in) :: hftable(cols,tabsize)
        !! Table halo mass function, log(dn/dlnm) and bias, log(b) as function 
        !! of log(halo mass)

        integer(c_int), intent(in), value :: cols
        !! Number of columns in mass-function table. It should be 2 or 3, if 
        !! integration weights are specified in the 3-rd column.

        integer(c_int64_t), intent(in), value :: tabsize
        !! Size of the mass-function table.

        real(c_double) :: res, res1, res2, hod(5)
        integer(c_int) :: use_weight

        hod = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ] ! HOD parameters

        use_weight = 0_c_int
        if ( cols < 3 ) then
            write (stderr, '("error: average_galaxy_bias: ",a)') 'insufficient data'
            res = -1._c_double
            return
        else if ( cols > 3 ) then
            use_weight = 1_c_int ! use last column as weights
        end if

        ! Average central galaxy density:
        call halo_average('c', use_weight, 0_c_int, hod, cols, tabsize, hftable, res1)
        
        ! Average satellite galaxy density:
        call halo_average('s', use_weight, 0_c_int, hod, cols, tabsize, hftable, res2)
        
        ! Bias integral
        call halo_average('t', use_weight, 1_c_int, hod, cols, tabsize, hftable, res )
        
        res = res / (res1 + res2)
        
    end function average_galaxy_bias

end module halo_model_mod

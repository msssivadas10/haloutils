module model_optimization_mod
    !! Optimizing halo model parameters based on observed values.
    !! WARNING: not tested, may remove later.

    use iso_c_binding
    use iso_fortran_env, only: stderr=>error_unit
    use halo_model_mod, only: halo_average, hmargs_t
    implicit none

    private
    public :: optimize_model
    
contains

    function cost_(hod, ngal_target, fsat_target, cols, tabsize, hftable) result(res)
        !! Cost function for HOD optimization 
        
        integer(c_int)    , intent(in) :: cols
        integer(c_int64_t), intent(in) :: tabsize
        real(c_double)    , intent(in) :: hod(5), ngal_target, fsat_target
        real(c_double)    , intent(in) :: hftable(cols,tabsize)
        
        real(c_double) :: res, res1, res2, ngal_test, fsat_test
        integer(c_int) :: use_weight

        use_weight = 0_c_int
        if ( cols > 2 ) use_weight = 1_c_int ! use last column as weights
    
        ! Average central galaxy density:
        call halo_average('c', use_weight, 0_c_int, hod, cols, tabsize, hftable, res1)
        
        ! Average satellite galaxy density:
        call halo_average('s', use_weight, 0_c_int, hod, cols, tabsize, hftable, res2)
        
        ngal_test = res1 + res2      ! Average total galaxy density
        fsat_test = res2 / ngal_test ! Average satellite fraction

        ! Cost: 
        res = (ngal_test / ngal_target - 1)**2 + (fsat_test / fsat_target - 1)**2 

    end function cost_

    subroutine optimize_model(args, ngal_target, fsat_target, bounds, m0_is_mmin,  &
                              cols, tabsize, hftable, abstol, maxsteps, stat       &
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

        integer(c_int), intent(in), value :: m0_is_mmin
        !! If non-zero, then set the value of `lnm0` to `lnm_min`.  

        real(c_double), intent(in) :: hftable(cols,tabsize)
        !! Table of halo mass function, log(dn/dlnm) as function of log(halo mass)

        integer(c_int), intent(in), value :: cols
        !! Number of columns in mass-function table. It should be 2 or 3, if 
        !! integration weights are specified in the 3-rd column.

        integer(c_int64_t), intent(in), value :: tabsize
        !! Size of the mass-function table.

        real(c_double), intent(in), value :: abstol
        !! Absolute tolerance value for checking convergence

        integer(c_int64_t), intent(in), value :: maxsteps
        !! Maximum number of steps for optimzation

        integer(c_int), intent(out) :: stat
        !! Status (0=success, 1=not converged)

        real(c_double), parameter :: c1   = 1.e-04_c_double
        real(c_double), parameter :: beta = 0.5_c_double
        
        integer(c_int)     :: i, fixed_param(5)
        integer(c_int64_t) :: k
        real(c_double)     :: lower_bound(5), upper_bound(5), x(5), trial_x(5)
        real(c_double)     :: grad(5), xa, xb, cost, step, trial_cost, g2
        
        stat = 1
        if ( cols < 2 ) then
            write (stderr, '("error: optimize_model: ",a)') 'insufficient data'
            return
        end if

        ! Correcting the lower and upper bounds: setting the bounds for mass 
        ! parameters (M_min, M0 or M1) to be -6 <= log(x) <= 20 and the width
        ! range to be [0, 10^4], alpha range to be [-10, 10]. If bounds are 
        ! specified, they should be in this range. 
        lower_bound = max( bounds(1:5, 1), [ -6.d0, 0.d0  , -6.d0, -6.d0, -10.d0 ] )
        upper_bound = min( bounds(1:5, 2), [ 20.d0, 1.d+04, 20.d0, 20.d0,  10.d0 ] )
        fixed_param = merge( &
            [1, 1, 1, 1, 1], &
            [0, 0, 0, 0, 0], &
            abs(lower_bound - upper_bound) < 1e-08_c_double &
        )

        x = [ args%lnm_min, args%sigma_m, args%lnm0, args%lnm1, args%alpha ]
        x = min( max(lower_bound, x), upper_bound ) ! clipping between bounds
        if ( m0_is_mmin /= 0 ) x(3) = x(1) ! set m_min and m0 parameters the same

        ! Evaluate initial cost
        cost = cost_(x, ngal_target, fsat_target, cols, tabsize, hftable)

        do k = 1, maxsteps

            x = min( max(lower_bound, x), upper_bound ) ! clipping between bounds
            
            ! Calculating the gradients numerically
            do i = 1, 5
                if ( fixed_param(i) /= 0 ) then
                    ! This parameter is fixed, so the gradient is set to zero, so 
                    ! that no variation is applied along that direction.
                    grad(i) = 0._c_double
                else if ( m0_is_mmin /= 0 .and. i == 3 ) then 
                    ! Since m0=m_min is forced, use the calculated gradient for 
                    ! parameter m_min for m0:
                    grad(3) = grad(1)
                else
                    trial_x(1:5) = x(1:5)
                    step = max( 1e-04_c_double, 1e-06_c_double*abs( x(i) ) )
                    
                    ! -- For left point
                    xa         = max( lower_bound(i), x(i) - step )
                    trial_x(i) = xa
                    if ( m0_is_mmin /= 0 ) trial_x(3) = trial_x(1) ! set M0 = Mmin
                    grad(i)    = cost_(trial_x, ngal_target, fsat_target, &
                                       cols, tabsize, hftable             &
                    )
                    ! -- For right point
                    xb         = min( upper_bound(i), x(i) + step )
                    trial_x(i) = xb
                    if ( m0_is_mmin /= 0 ) trial_x(3) = trial_x(1) ! set M0 = Mmin
                    trial_cost = cost_(trial_x, ngal_target, fsat_target, &
                                       cols, tabsize, hftable             &
                    )
                    grad(i)    = trial_cost - grad(i)
                    
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
               trial_x = x - step * grad 
               trial_x = min( max(lower_bound, trial_x), upper_bound ) ! clipping between bounds
               if ( m0_is_mmin /= 0 ) trial_x(3) = trial_x(1) ! set M0 = Mmin
               trial_cost = cost_(trial_x, ngal_target, fsat_target, &
                                  cols, tabsize, hftable             &
                ) 
                if ( trial_cost < cost - c1*step*g2 ) then
                    exit
                else
                    step = beta*step
                    if ( step < 1e-12_c_double ) exit 
                end if
            end do

            ! Accepting the trial values
            x    = trial_x
            cost = trial_cost

        end do

    end subroutine optimize_model
    
end module model_optimization_mod
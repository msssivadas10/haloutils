module halo_bias_mod
    !! Linear halo bias models.
    !!
    !! NOTE: Since halo bias models also use the struct `hmfargs_t` to 
    !! hold its arguments, first call the setup routine for the halo mass
    !! function model, and then call the bias setup routine (if any)

    use iso_c_binding
    use constants_mod
    use mass_function_mod
    implicit none

    private
    public :: hbf_cole89, hbf_sheth01, hbf_tinker10        
    
contains

! Cole & Kaiser (1989)

    function hbf_cole89(args) result(res) bind(c)
        !! Calculate the linear bias using model given by Cole & Kaiser (1989) 
        !! and  Mo & White (1996).
        !!
        !! References:
        !! - Shaun Cole and Nick Kaiser. Biased clustering in the cold dark 
        !!   matter cosmogony. Mon. Not.R. astr. Soc. 237, 1127-1146 (1989).
        !! - H. J. Mo, Y. P. Jing and S. D. M. White. High-order correlations  
        !!   of peaks and haloes: a step towards understanding galaxy biasing. 
        !!   Mon. Not. R. Astron. Soc. 284, 189-201 (1997).

        type(hmfargs_t), intent(inout) :: args
        !! Arguments
        
        real(c_double) :: res, nu
        
        ! Cole & Kaiser (1989) model b(nu)
        nu  = delta_sc / args%s
        res = 1 + ( nu**2 - 1 ) / delta_sc
        
    end function hbf_cole89

! Sheth et al. (2001)

    function hbf_sheth01(args) result(res) bind(c)
        !! Calculate the linear bias using model given by Sheth et al. (2001).
        !!
        !! References:
        !! - Jeremy L. Tinker et al. The large scale bias of dark matter halos:  
        !!   Numerical calibration and model tests. <http://arxiv.org/abs/1001.3162v2> 
        !!   (2010)
       
        type(hmfargs_t), intent(inout) :: args
        !! Arguments
        
        real(c_double) :: res, nu, sqrt_a, anu2, c1
        
        real(c_double), parameter :: a = 0.707_c_double ! Model parameter a
        real(c_double), parameter :: b = 0.5_c_double   ! Model parameter b
        real(c_double), parameter :: c = 0.6_c_double   ! Model parameter c
        
        ! Sheth et al. (2001) model b(nu)
        nu     = delta_sc / args%s
        sqrt_a = sqrt(a)
        anu2   = a * nu**2
        c1     = (1._c_double - c)
        res    = ( anu2**c + b * c1 * (1 - 0.5_c_double*c) )
        res    = sqrt_a * anu2 + sqrt_a * b * anu2**c1 - anu2**c / res
        res    = 1 + 1._c_double / sqrt_a / delta_sc * res
        
    end function hbf_sheth01

! Tinker et al (2010)

    function hbf_tinker10(args) result(res) bind(c)
        !! Calculate the linear bias using model given by Tinker et al. (2008).
        !!
        !! References:
        !! - Jeremy L. Tinker et al. The large scale bias of dark matter halos:  
        !!   Numerical calibration and model tests. <http://arxiv.org/abs/1001.3162v2> 
        !!   (2010)
       
        type(hmfargs_t), intent(inout) :: args
        !! Arguments
        
        real(c_double) :: res, nu, A_, B_, C_, a, b, c, x, y

        y = log10( args%Delta_m )
        x = exp( -( 4._c_double / y )**4 )

        A_ = 1 + 0.24_c_double * y * x       ! Parameter A
        a  = 0.44_c_double*y - 0.88_c_double ! Parameter a

        B_ = 0.183_c_double ! Parameter B
        b  = 1.5_c_double   ! Parameter b
        
        C_ = 0.019_c_double + 0.107_c_double*y + 0.19_c_double*x ! Parameter C
        c  = 2.4_c_double ! Parameter c
        
        ! Tinker et al (2010) model b(nu)
        nu  = delta_sc / args%s 
        res = 1._c_double - A_*nu**a / ( nu**a + delta_sc**a ) + B_*nu**b + C_*nu**c
        
    end function hbf_tinker10
    
end module halo_bias_mod
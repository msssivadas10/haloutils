module mass_function_mod
    !! Halo mass-function models.
    
    use iso_c_binding
    use constants_mod
    use rfunctions_mod
    implicit none

    private
    public :: setup_hmf, setup_hmf_tinker08, hmf_press74, hmf_sheth01, &
              hmf_jenkins01, hmf_reed03, hmf_tinker08, hmf_courtin10,  &
              hmf_crocce10              
    
    integer, parameter :: dp = c_double

    real(c_double), parameter :: T08(5, 9) = reshape([ &
    !   Delta   ,    A       ,    a       ,    b       ,    c
         200._dp,    0.186_dp,    1.470_dp,    2.570_dp,    1.190_dp, & 
         300._dp,    0.200_dp,    1.520_dp,    2.250_dp,    1.270_dp, & 
         400._dp,    0.212_dp,    1.560_dp,    2.050_dp,    1.340_dp, & 
         600._dp,    0.218_dp,    1.610_dp,    1.870_dp,    1.450_dp, &
         800._dp,    0.248_dp,    1.870_dp,    1.590_dp,    1.580_dp, &
        1200._dp,    0.255_dp,    2.130_dp,    1.510_dp,    1.800_dp, &
        1600._dp,    0.260_dp,    2.300_dp,    1.460_dp,    1.970_dp, &
        2400._dp,    0.260_dp,    2.530_dp,    1.440_dp,    2.240_dp, &
        3200._dp,    0.260_dp,    2.660_dp,    1.410_dp,    2.440_dp  &
    ], shape(T08))  
    !! Table of Tinker (2008) mass-function parameters as function of 
    !! overdensity Delta (w.r.to mean background density) 

    type, public, bind(c) :: hmfargs_t
        !! A struct containing values of various arguments for halo 
        !! mass-function calculation routines.

        real(c_double) :: z        !! Redshift
        real(c_double) :: lnm      !! Natural log of halo mass in Msun
        real(c_double) :: H0       !! Hubble parameter
        real(c_double) :: Om0      !! Total matter density parameter
        real(c_double) :: Delta_m  !! Matter overdensity w.r.to mean background density
        real(c_double) :: s        !! Matter variance corresponding to halo mass
        real(c_double) :: dlnsdlnm !! Log derivative of matter variance w.r.to halo mass
        real(c_double) :: rho_m    !! Total matter density at redshift 0 (unit: Msun/Mpc^3)

        real(c_double) :: param(16)
        !! Model parameters: since this is used by both mass-function and 
        !! halo bias models, first 8 values are mass reserved for mass-function
        !! parameters and the other 8 are for bias parameters. 
        
    end type
    
contains

! Basic setup routine

    subroutine setup_hmf(args, filt, pktab, size, cls) bind(c)
        !! Calculate related quantities for halo mass-function calculation.
        !!
        !! NOTE: If a model has specific setup routine, that should be called   
        !! instead of this. These are:
        !! - `setup_hmf_tinker08` - Tinker (2008) 

        type(hmfargs_t), intent(inout) :: args

        integer(c_int), intent(in), value :: filt
        !! Code of filter function (0=tophat, 1=gaussian)

        real(c_double), intent(in) :: pktab(cls, size)
        !! Precalculated power spectrum table. The columns should be
        !! 1=Nodes of integration (natural log of k in 1/Mpc), 
        !! 2=Value of natural log of power spectrum, 
        !! 3=Weights for integration.
        
        integer(c_int64_t), intent(in), value :: size
        !! Size of the power spectrum table: must be a multiple of 7.

        integer(c_int), intent(in), value :: cls
        !! Columns of the power spectrum table: must be 2 or 3. If 3, use the 
        !! the last column as weights. Otherwise, Simpson's rule is used. For 
        !! other values, return value will be NaN.

        real(c_double) :: lnr, rho_h, s2

        args%rho_m = args%Om0 * ( critical_density_const * args%H0**2 ) ! matter density at z=0 in Msun/Mpc^3 

        ! Lagrangian radius (r) corresponding to halo mass
        rho_h = args%rho_m ! * args%Delta_m
        lnr   = ( args%lnm + log(3._c_double / (4*pi) / rho_h ) ) / 3._c_double ! r in Mpc

        ! Matter variance
        s2     = variance(lnr, 0, 0, filt, pktab, size, cls)
        args%s = sqrt(s2)
        
        ! Log derivative of matter variance
        args%dlnsdlnm = variance(lnr, 0, 1, filt, pktab, size, cls) ! ds2/dr
        args%dlnsdlnm = args%dlnsdlnm * ( exp(lnr) / s2 / 6._c_double )
        
    end subroutine setup_hmf

    function convert_fs_to_hmf(args, fs, target_code) result(res)
        !! Calculate the value of halo mass-function given f(s) and other values.

        type(hmfargs_t), intent(in) :: args
        !! Arguments

        real(c_double), intent(in), value :: fs
        !! Value of mass-function f(s)

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m)
        
        real(c_double) :: res, m

        m = exp(args%lnm) ! Halo mass in Msun
        
        res = fs * abs(args%dlnsdlnm) * args%rho_m / m ! dn/dlnm in Mpc^-3
        select case ( target_code )
        case ( 2 )
            res = res / log(10._c_double) ! dn/dlog10m
        case ( 0 ) 
            res = res / m ! dn/dm
        end select

    end function convert_fs_to_hmf

! -- Implementations of Some Halo Mass-function Models -- !

! Press & Schechter (1974)

    function hmf_press74(args, target_code) result(res) bind(c)
        !! Calculate halo mass-massfunction using Press & Schechter (1974) model.
        !! This is based on spherical collapse model.
        
        type(hmfargs_t), intent(in) :: args
        !! Arguments

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m, other=fs)

        real(c_double) :: res, nu

        ! Press & Schechter (1974) model f(s)
        nu  = delta_sc / args%s
        res = sqrt(2._c_double / pi) * nu * exp(-0.5_c_double * nu**2)

        if ( target_code == 0 .or. target_code == 1 .or. target_code == 2 ) &
            res = convert_fs_to_hmf(args, res, target_code)
        
    end function hmf_press74

! Sheth et al (2001)

    function hmf_sheth01(args, target_code) result(res) bind(c)
        !! Calculate halo mass-massfunction using Sheth et al (2001) model.
        !! This is based on ellipsoidal collapse model.
        
        type(hmfargs_t), intent(in) :: args
        !! Arguments

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m, other=fs)

        real(c_double) :: res, nu

        real(c_double), parameter :: A_ = 0.3222_c_double !! Model parameter A 
        real(c_double), parameter :: a  = 0.707_c_double  !! Model parameter a
        real(c_double), parameter :: p  = 0.3_c_double    !! Model parameter p
        
        ! Sheth et al (2001) model f(s)
        nu  = delta_sc / args%s
        res = A_ * sqrt(2*a / pi) * nu * exp(-0.5_c_double*a*nu**2) * (1 + (nu**2 / a)**(-p))

        if ( target_code == 0 .or. target_code == 1 .or. target_code == 2 ) &
            res = convert_fs_to_hmf(args, res, target_code)
        
    end function hmf_sheth01

! Jenkins et al (2001)

    function hmf_jenkins01(args, target_code) result(res) bind(c)
        !! Calculate halo mass-massfunction using Jenkins et al (2001) model.
        !! This model is valid for the range -1.2 <= -log(sigma) <= 1.05.
        !!
        !! References:
        !! - A. Jenkins et al. The mass function of dark matter halos. <http://arxiv.org/abs/astro-ph/0005260v2>
        
        type(hmfargs_t), intent(in) :: args
        !! Arguments

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m, other=fs)

        real(c_double) :: res, s

        ! Jenkins et al (2001) models f(s)
        s   = args%s
        res = 0.315_c_double * exp( -abs( 0.61_c_double - log(s) )**3.8_c_double )

        if ( target_code == 0 .or. target_code == 1 .or. target_code == 2 ) &
            res = convert_fs_to_hmf(args, res, target_code)
        
    end function hmf_jenkins01

! Reed et al (2003)

    function hmf_reed03(args, target_code) result(res) bind(c)
        !! Calculate halo mass-massfunction using Reed et al (2003) model. 
        !!
        !! References:
        !! - Zarija Lukić et al. The halo mass function: high-redshift evolution 
        !!   and universality. <http://arXiv.org/abs/astro-ph/0702360v2>.

        type(hmfargs_t), intent(in) :: args
        !! Arguments

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m, other=fs)

        real(c_double) :: res, s

        ! Reed et al (2003) model f(s)
        s   = args%s
        res = exp(-0.7_c_double / s / cosh(2*s)**2) * hmf_sheth01(args, -1)

        if ( target_code == 0 .or. target_code == 1 .or. target_code == 2 ) &
            res = convert_fs_to_hmf(args, res, target_code)
        
    end function hmf_reed03

! Warren et al (2006)

    function hmf_warren06(args, target_code) result(res) bind(c)
        !! Calculate halo mass-massfunction using Warren et al (2006) model. 
        !!
        !! References:
        !! - Zarija Lukić et al. The halo mass function: high-redshift evolution 
        !!   and universality. <http://arXiv.org/abs/astro-ph/0702360v2>.

        type(hmfargs_t), intent(in) :: args
        !! Arguments

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m, other=fs)

        real(c_double) :: res, s

        real(c_double), parameter :: A_ = 0.7234_c_double !! Model parameter A
        real(c_double), parameter :: a  = 1.6250_c_double !! Model parameter a
        real(c_double), parameter :: b  = 0.2538_c_double !! Model parameter b
        real(c_double), parameter :: c  = 1.1982_c_double !! Model parameter c

        ! Warren et al (2006) model f(s)
        s   = args%s
        res = A_ * ( s**(-a) + b ) * exp(-c / s**2)

        if ( target_code == 0 .or. target_code == 1 .or. target_code == 2 ) &
            res = convert_fs_to_hmf(args, res, target_code)
        
    end function hmf_warren06

! Tinker (2008)

    subroutine setup_hmf_tinker08(args, filt, pktab, size, cls) bind(c)
        !! Calculate related quantities for Tinker (2008) halo mass-function 
        !! calculation.
        !!
        !! References: 
        !! - Jeremy Tinker et al. Toward a halo mass function for precision cosmology:  
        !!   The limits of universality. <http://arXiv.org/abs/0803.2706v1> (2008).


        type(hmfargs_t), intent(inout) :: args

        integer(c_int), intent(in), value :: filt
        !! Code of filter function (0=tophat, 1=gaussian)

        real(c_double), intent(in) :: pktab(cls, size)
        !! Precalculated power spectrum table. The columns should be
        !! 1=Nodes of integration (natural log of k in 1/Mpc), 
        !! 2=Value of natural log of power spectrum, 
        !! 3=Weights for integration.
        
        integer(c_int64_t), intent(in), value :: size
        !! Size of the power spectrum table: must be a multiple of 7.

        integer(c_int), intent(in), value :: cls
        !! Columns of the power spectrum table: must be 2 or 3. If 3, use the 
        !! the last column as weights. Otherwise, Simpson's rule is used. For 
        !! other values, return value will be NaN.

        real(c_double) :: slope(4)
        integer :: k, klo, khi

        ! Basic setup
        call setup_hmf(args, filt, pktab, size, cls)

        ! Get the values for parameters A0, a0, b0, c0 and c0: linear 
        ! interpolate from the table.
        klo = 1
        khi = 9
        do while( khi - klo > 1 )
            k = (khi + klo) / 2
            if ( T08(1,k) > args%Delta_m ) then
                khi = k
            else
                klo = k
            end if
        end do
        slope = (T08(2:5,khi) - T08(2:5,klo)) / (T08(1,khi) - T08(1,klo))
        args%param(1:4) = T08(2:5,klo) + slope * (args%Delta_m - T08(1,klo))

        ! Parameter alpha
        args%param(5) = -( 0.75_c_double / log10( args%Delta_m / 75._c_double ) )**1.2_c_double 
        args%param(5) =  10._c_double**args%param(5)
        
    end subroutine setup_hmf_tinker08

    function hmf_tinker08(args, target_code) result(res) bind(c)
        !! Calculate halo mass-massfunction using Tinker (2008) model.
        !!
        !! References: 
        !! - Jeremy Tinker et al. Toward a halo mass function for precision cosmology:  
        !!   The limits of universality. <http://arXiv.org/abs/0803.2706v1> (2008).
        
        type(hmfargs_t), intent(in) :: args
        !! Arguments

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m, other=fs)
        
        real(c_double) :: res, A_, a, b, c, alpha, zp1, s

        zp1 = args%z + 1
        
        ! Parameter `A`
        A_ = args%param(1) * zp1**(-0.14_c_double ) 
        
        ! Parameter `a`  
        a = args%param(2) * zp1**(-0.06_c_double ) 
        
        ! Parameter `b`
        alpha = args%param(5)
        b     = args%param(3) * zp1**(-alpha)
        
        ! Parameter `c`
        c = args%param(4)
        
        ! Tinker (2008) model f(s)
        s   = args%s
        res = A_ * (1._c_double + (b / s)**a) * exp(-c / s**2)

        if ( target_code == 0 .or. target_code == 1 .or. target_code == 2 ) &
            res = convert_fs_to_hmf(args, res, target_code)
        
    end function hmf_tinker08

! Courtin et al (2010)

    function hmf_courtin10(args, target_code) result(res) bind(c)
        !! Calculate halo mass-massfunction using Courtin et al (2010) model.
        !!
        !! References:
        !! - J. Courtin et al. Imprints of dark energy on cosmic structure  
        !!   formation-II. Non-universality of the halo mass function. 
        !!   Mon. Not. R. Astron. Soc. 410, 1911-1931 (2011)
        
        type(hmfargs_t), intent(in) :: args
        !! Arguments

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m, other=fs)
        
        real(c_double) :: res, nu

        real(c_double), parameter :: A_ = 0.348_c_double !! Model parameter A 
        real(c_double), parameter :: a  = 0.695_c_double !! Model parameter a
        real(c_double), parameter :: p  = 0.1_c_double   !! Model parameter p
        
        ! Courtin et al (2010) model f(s)
        nu  = delta_sc / args%s
        res = A_ * sqrt(2*a / pi) * nu * exp(-0.5_c_double*a*nu**2) * (1 + (nu**2 / a)**(-p))

        if ( target_code == 0 .or. target_code == 1 .or. target_code == 2 ) &
            res = convert_fs_to_hmf(args, res, target_code)
        
    end function hmf_courtin10
    
! Crocce et al (2010)

    function hmf_crocce10(args, target_code) result(res) bind(c)
        !! Calculate halo mass-massfunction using Courtin et al (2010) model.
        !!
        !! References:
        !! - Martín Crocce et al. Simulating the Universe with MICE : The  
        !!   abundance of massive clusters. <http://arxiv.org/abs/0907.0019v2>
        
        type(hmfargs_t), intent(in) :: args
        !! Arguments

        integer(c_int), intent(in), value :: target_code
        !! Code for output value (0=dn/dm, 1=dn/dlnm, 2=dn/dlog10m, other=fs)
        
        real(c_double) :: res, s, zp1, A_, a, b, c

        ! Redshift depenedent parameters:
        zp1 = args%z + 1
        A_  = 0.580_c_double * zp1**( -0.130_c_double ) ! A
        a   = 1.370_c_double * zp1**( -0.150_c_double ) ! a
        b   = 0.300_c_double * zp1**( -0.084_c_double ) ! b
        c   = 1.036_c_double * zp1**( -0.024_c_double ) ! c

        ! Crocce et al (2010) model f(s)
        s   = args%s
        res = A_ * ( s**(-a) + b ) * exp(-c / s**2)

        if ( target_code == 0 .or. target_code == 1 .or. target_code == 2 ) &
            res = convert_fs_to_hmf(args, res, target_code)
        
    end function hmf_crocce10
    
end module mass_function_mod
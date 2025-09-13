module power_spectrum_mod
    !! Linear matter power spectrum models.

    use iso_c_binding
    implicit none

    integer, parameter :: dp = c_double

    real(c_double), parameter :: e = 2.718281828459045_dp 
    !! Base of natural logarithm, e
    
    type, bind(c) :: psargs_t
        !! A struct containing values of various arguments and parameters required
        !! for matter power spectrum calculations.

        ! -- Parameters set by the user -- !
        real(c_double) :: z          !! Redshift 
        real(c_double) :: h          !! Hubble parameter (in unit of 100 km/sec/Mpc)
        real(c_double) :: Omh2       !! Matter density parameter 
        real(c_double) :: Obh2       !! Baryon matter density parameter 
        real(c_double) :: Onuh2      !! Massive neutrino density parameter 
        real(c_double) :: Nnu        !! Number of massive neutrino species
        real(c_double) :: ns         !! Index of the initial power spectrum
        real(c_double) :: sigma8     !! Matter variance at scale 8 Mpc/h: to normalize power spectrum
        real(c_double) :: theta      !! Temperature of the cosmic microwave background (in unit of 2.7 K)
        real(c_double) :: dplus_z    !! Linear growth at this redshift 
        real(c_double) :: dplus_0    !! Linear growth at redshift 0 (used for normalising the growth factor)
        real(c_double) :: norm       !! Power spectrum normalization for sigma8=1
        integer(c_int) :: include_nu !! Flag indicating if to include neutrino in the transfer function

        ! -- Parameters to be calculated by init subroutines -- !
        real(c_double) :: z_eq     !! Redshift corresponding to the matter - radiation equality epoch
        real(c_double) :: z_d      !! Redshift corresponding to the drag epoch
        real(c_double) :: s        !! Sound horizon in Mpc (see Eqn 26 in the EH paper)
        real(c_double) :: k_silk   !! Silk damping scale  (see Eqn 7 in the EH paper)
        real(c_double) :: param(5) !! Various parameters for the model 
        
    end type
    
contains

! Eisenstein & Hu (1998) Model for Zero / Trace Baryon Content
    
    subroutine init_eisenstein98_zb(args) bind(c)
        !! Initialize parameters related to Eisenstein & Hu matter power spectrum
        !! for zero baryon.
        
        type(psargs_t), intent(inout) :: args

        real(c_double) :: Omh2, Obh2, theta, c1, c2, fb

        Omh2  = args%Omh2
        Obh2  = args%Obh2
        theta = args%theta

        ! Redshift at matter-radiation equality (Eqn. 1)
        args%z_eq = 2.5e+04_dp * Omh2 / theta**4

        ! Redshift at drag epoch (Eqn. 2)
        c1  = 0.313_dp*(1 + 0.607_dp*Omh2**0.674_dp) / Omh2**0.419_dp
        c2  = 0.238_dp*Omh2**0.223_dp
        args%z_d = 1291.0_dp*(Omh2**0.251_dp)*(1 + c1*Obh2**c2) / (1 + 0.659_dp*Omh2**0.828_dp)

        ! Sound horizon (Eqn. 26)
        args%s = 44.5_dp*log( 9.83_dp/Omh2 ) / sqrt( 1 + 10._dp*Obh2**0.75_dp )

        ! Parameter alpha_Gamma, Eqn. 31
        fb = Obh2 / Omh2
        args%param(1) = 1._dp - 0.328_dp*log( 431*Omh2 ) * fb + 0.38_dp*log( 22.3_dp*Omh2 ) * fb**2

    end subroutine init_eisenstein98_zb

    function ps_eisenstein98_zb(lnk, args) result(retval) bind(c)
        !! Return the value of Eisenstein & Hu matter power spectrum for zero baryon.
        !!
        !! NOTE: call only after initialization with `init_eisenstein98_zb`
        
        real(c_double), intent(in), value :: lnk
        !! Natural log of wavenumber in 1/Mpc

        type(psargs_t), intent(in) :: args
        !! Other arguments

        real(c_double) :: retval 
        !! Natural log of matter power spectrum value

        real(c_double) :: Omh2, theta, alpha_g, gamma_eff, dplus_z, s, q, k, t2, t3

        Omh2    = args%Omh2
        theta   = args%theta 
        alpha_g = args%param(1)
        s       = args%s
        
        ! Linear growth factor, normalized so that value = 1 at redshift 0
        dplus_z = args%dplus_z / args%dplus_0 

        k = exp(lnk) ! Mpc^-1

        ! Shape parameter, Eqn 30
        gamma_eff = Omh2 * ( alpha_g + ( 1 - alpha_g ) / ( 1 + ( 0.43_dp*k*s )**4 ) ) 
        q = k * ( theta**2 / gamma_eff ) 
        
        ! Transfer function
        t2 = log( 2*e + 1.8_dp*q )
        t3 = 14.2_dp + 731.0_dp / ( 1 + 62.5_dp*q )
        retval = t2 / (t2 + t3*q**2)

        ! Interpolation using growth factor
        retval = dplus_z * retval 

        ! Linear matter power spectrum
        retval = (args%sigma8**2 * args%norm) * retval**2 * k**( args%ns )
        retval = log(retval)
        
    end function ps_eisenstein98_zb

! Eisenstein & Hu (1998) Model for Non-zero Massive Neutrino Content

    subroutine init_eisenstein98_mnu(args) bind(c)
        !! Initialize parameters related to Eisenstein & Hu matter power spectrum
        !! including massive neutrino.

        type(psargs_t), intent(inout) :: args

        real(c_double) :: Omh2, Obh2, Onuh2, Nnu, fb, fnu, fnb, fc, fcb, pc, pcb, &
                        c1, c2, yd

        Omh2  = args%Omh2
        Obh2  = args%Obh2
        Onuh2 = args%Onuh2
        Nnu   = args%Nnu
        fb    = Obh2  / Omh2 ! Baryon fraction     
        fnu   = Onuh2 / Omh2 ! Massive neutrino fraction
        fnb   = fnu + fb     ! Massive neutrino + baryon    
        fc    = 1._dp - fnb  ! CDM fraction    
        fcb   = fc + fb      ! CDM + baryon fraction

        ! Redshift at matter-radiation equality (eqn. 1)
        args%z_eq = 2.5e+04_dp * Omh2 / args%theta**4 

        ! Redshift at drag epoch (eqn. 2)
        c1  = 0.313_dp*(1 + 0.607_dp*Omh2**0.674_dp) / Omh2**0.419_dp
        c2  = 0.238_dp*Omh2**0.223_dp
        args%z_d = 1291.0_dp*(Omh2**0.251_dp)*(1 + c1*Obh2**c2) / (1 + 0.659_dp*Omh2**0.828_dp)

        ! Sound horizon (eqn. 26)
        args%s = 44.5_dp*log( 9.83_dp/Omh2 ) / sqrt( 1._dp + 10._dp*Obh2**0.75_dp ) 

        ! Eqn. 14 in EH98 paper
        pc  = 0.25_dp*( 5._dp - sqrt( 1._dp + 24.0_dp*fc  ) ) 
        pcb = 0.25_dp*( 5._dp - sqrt( 1._dp + 24.0_dp*fcb ) )

        ! Small-scale suppression, alpha_nu (Eqn. 15)
        yd  = (1 + args%z_eq) / (1 + args%z_d) ! eqn. 3
        args%param(1) = (fc / fcb) * (5 - 2*(pc + pcb)) / (5 - 4*pcb)   &                                                 
            * (1 - 0.533_dp*fnb + 0.126_dp*fnb**3)                      &
            / (1 - 0.193_dp*sqrt(fnu*Nnu) + 0.169_dp*fnu*Nnu**0.2_dp)   &
            * (1 + yd)**(pcb - pc)                                      &                                        
            * (1 + 0.5_dp*(pc - pcb) * (1 + 1._dp / (3 - 4*pc) / (7 - 4*pcb)) * (1 + yd)**(-1))
        
        ! Eqn. 21, beta_c
        args%param(2) = (1 - 0.949_dp*fnb)**(-1)

        ! Parameter related to growth factor suppression (y_fs / q^2) (eqn. 14)
        args%param(3) = 17.2_dp*fnu*( 1 + 0.488_dp*fnu**(-7._dp/6._dp) ) * (Nnu/fnu)**2

        ! Constant part of B_k in Eqn. 22
        args%param(4) = 1.2_dp*fnu**0.64_dp * Nnu**(0.3_dp + 0.6_dp*fnu)

    end subroutine init_eisenstein98_mnu

    function ps_eisenstein98_mnu(lnk, args) result(retval) bind(c)
        !! Return the value of Eisenstein & Hu matter power spectrum including massive
        !! neutrinos.
        !!
        !! NOTE: call only after initialization with `init_eisenstein98_mnu`

        real(c_double), intent(in), value :: lnk
        !! Natural log of wavenumber in 1/Mpc

        type(psargs_t), intent(in) :: args
        !! Other arguments

        real(c_double) :: retval
        !! Natural log of matter power spectrum value

        real(c_double) :: Omh2, anu, dplus_z, dplus_c, beta_c, fnu, fcb, pcb,   &
                        Bconst, yfs_q2, s, q, k, t2, t3

        Omh2   = args%Omh2
        anu    = sqrt(args%param(1))
        beta_c = args%param(2)
        yfs_q2 = args%param(3)
        Bconst = args%param(4)
        s      = args%s
        fnu    = args%Onuh2 / Omh2 ! Massive neutrino fraction
        fcb    = 1 - fnu           ! CDM + baryon fraction
        pcb    = 0.25_dp*( 5._dp - sqrt( 1._dp + 24.0_dp*fcb ) )
        
        k = exp(lnk) ! wavenumber in Mpc^-1
        q = k * ( args%theta**2 / Omh2 ) ! dimension-less wavenumber

        ! Linear growth factor
        dplus_c = 2.5_dp*( Omh2 / args%h**2 )*(args%z_eq + 1) ! Used as normalization factor for growth
        dplus_z = dplus_c * args%dplus_z

        ! Suppressed growth factor
        t2  = yfs_q2 * q**2 ! y_fs in eqn. 14
        if ( args%include_nu /= 0 ) then 
            ! Dcbnu (eqn., 12), including neutrino
            t2 = ( fcb**(0.7_dp/pcb) + dplus_z / (1 + t2) )**(pcb/0.7_dp) * dplus_z**(1 - pcb) 
        else 
            ! Dcb (eqn., 13), not incl. neutrino
            t2 = ( (1 + dplus_z) / (1 + t2) )**(pcb/0.7_dp) * dplus_z**(1 - pcb) 
        end if
        retval = t2 / dplus_z

        ! Master function T_master (Eqn. 22-24)
        q  = 3.92_dp*q*sqrt(args%Nnu) / fnu                 ! q_nu in Eqn. 23
        t2 = 1 + Bconst / (q**(-1.6_dp) + q**0.8_dp) ! B_k in Eqn. 22
        retval = retval * t2   

        ! Transfer function T_sup (Eqn. 17-20)
        t2 = Omh2*( anu + (1 - anu)/(1 + (0.43*k*s)**4) ) ! Shape parameter, Gamma_eff (Eqn. 16)
        q  = k * args%theta**2 / t2                       ! Effective wavenumber (Eqn. 17)
        t2 = log( e + 1.84_dp*beta_c * anu * q )          ! L (Eqn. 19)
        t3 = 14.4_dp + 325._dp / (1 + 60.5_dp*q**1.11_dp) ! C (Eqn. 20)
        retval = t2 / (t2 + t3*q**2)                      ! Eqn. 18

        ! Linear interpolation using growth factor
        dplus_z = args%dplus_z / args%dplus_0 ! normalized so that value = 1 at redshift 0
        retval  = retval * dplus_z  

        ! Linear matter power spectrum
        retval = (args%sigma8**2 * args%norm) * retval**2 * k**( args%ns )
        retval = log(retval)
        
    end function ps_eisenstein98_mnu

! Eisenstein & Hu (1998) Model with Baryon Acoustic Oscillation

    subroutine init_eisenstein98_bao(args) bind(c)
        !! Initialize parameters related to Eisenstein & Hu matter power spectrum
        !! including baryon acoustic oscillations (BAO).

        type(psargs_t), intent(inout) :: args

        real(c_double) :: Omh2, Obh2, fb, fc, c1, c2, k_eq, R_eq, R_d, y, Gy

        Omh2 = args%Omh2
        Obh2 = args%Obh2
        fb   = Obh2 / Omh2                  ! Baryon fraction
        fc   = 1 - fb - (args%Onuh2 / Omh2) ! CDM fraction

        ! Redshift at matter-radiation equality (eqn. 1)
        args%z_eq = 2.5e+04_dp * Omh2 / args%theta**4

        ! Redshift at drag epoch (eqn. 2)
        c1  = 0.313_dp*(1 + 0.607_dp*Omh2**0.674_dp) / Omh2**0.419_dp
        c2  = 0.238_dp*Omh2**0.223_dp
        args%z_d = 1291.0_dp*(Omh2**0.251_dp)*(1 + c1*Obh2**c2) / (1 + 0.659_dp*Omh2**0.828_dp)

        ! Scale of particle horizon at z_eq
        k_eq = 7.46e-02_dp*Omh2 / args%theta**2

        ! Ratio of baryon - photon momentum density (eqn. 5)
        R_eq = 31.5_dp*Obh2*args%theta**(-4) * (args%z_eq / 1.0e+03_dp)**(-1) ! at z_eq
        R_d  = 31.5_dp*Obh2*args%theta**(-4) * (args%z_d  / 1.0e+03_dp)**(-1) ! at z_d

        ! Sound horizon (eqn. 26)
        args%s = (2._dp / 3._dp / k_eq)    &
                    * sqrt(6._dp / R_eq)   &
                    * log((sqrt(1 + R_d) + sqrt(R_d + R_eq)) / (1 + sqrt(R_eq)))

        ! Silk damping scale (eqn. 7)
        args%k_silk = 1.6_dp*Obh2**0.52_dp * Omh2**0.73_dp * (1 + (10.4_dp*Omh2)**(-0.95_dp))

        ! Parameter alpha_c, Eqn. 11
        c1 = (46.9_dp*Omh2)**0.670_dp * (1._dp + (32.1_dp*Omh2)**(-0.532_dp))
        c2 = (12.0_dp*Omh2)**0.424_dp * (1._dp + (45.0_dp*Omh2)**(-0.582_dp))
        args%param(1) = c1**(-fb) * c2**(-fb**3)

        ! Parameter beta_c, Eqn. 12
        c1 = 0.944_dp*(1._dp + (458._dp*Omh2)**(-0.708_dp))**(-1)
        c2 = (0.395_dp*Omh2)**(-0.0266_dp)
        args%param(2) = (1._dp + c1*(fc**c2 - 1._dp))**(-1)

        ! Parameter alpha_b, Eqn. 14-15
        y  = (1 + args%z_eq) / (1 + args%z_d)
        Gy = y*( -6*sqrt(1 + y) + (2 + 3*y) * log( (sqrt(1 + y) + 1) / (sqrt(1 + y) - 1) ) )
        args%param(3) = 2.07_dp*k_eq * args%s * (1 + R_d)**(-0.75_dp) * Gy

        ! Parameter beta_b, Eqn. 24
        args%param(4) = 0.5_dp + fb + (3 - 2*fb)*sqrt((17.2_dp*Omh2)**2 + 1)

        ! Parameter beta_node, Eqn. 23
        args%param(5) = 8.41_dp*Omh2**0.435_dp

    end subroutine init_eisenstein98_bao

    function ps_eisenstein98_bao(lnk, args) result(retval) bind(c)
        !! Return the value of Eisenstein & Hu matter power spectrum including baryon
        !! acoustic oscillations (BAO).
        !!
        !! NOTE: call only after initialization with `init_eisenstein98_bao`

        real(c_double), intent(in), value :: lnk
        !! Natural log of wavenumber in 1/Mpc

        type(psargs_t), intent(in) :: args
        !! Other arguments

        real(c_double) :: retval
        !! Natural log of matter power spectrum value

        real(c_double) :: alpha_c, alpha_b, beta_c, beta_b, beta_node, dplus_z,   &
                        fc, fb, s, ks, q, k, t0b, t0ab, t2, t3, t4, f, st

        alpha_c   = args%param(1)
        alpha_b   = args%param(2)
        beta_c    = args%param(3) 
        beta_b    = args%param(4)
        beta_node = args%param(5)
        s         = args%s
        fb        = args%Obh2 / args%Omh2             ! Baryon fraction
        fc        = 1 - fb - (args%Onuh2 / args%Omh2) ! CDM fraction
        
        ! Linear growth factor, normalized so that value = 1 at redshift 0
        dplus_z = args%dplus_z / args%dplus_0 

        k  = exp(lnk) ! Mpc^-1
        q  = k * ( args%theta**2 / args%Omh2 ) 
        ks = k*s 

        ! Transfer function: CDM part (Eqn. 17-20)
        f    = 1._dp / (1 + (ks / 5.4_dp)**4)  ! Eqn. 18
        t2   = log( e + 1.8_dp*beta_c*q )
        t3   = 14.2_dp + 386.0_dp / (1 + 69.9_dp*q**1.08_dp)            ! Eqn. 20 with alpha_c=1
        t4   = 14.2_dp / alpha_c + 386.0_dp / (1 + 69.9_dp*q**1.08_dp ) ! Eqn. 20 with alpha_c
        t0b  = t2 / (t2 + t3*q**2)           ! Eqn. 19 
        t0ab = t2 / (t2 + t4*q**2)           ! Eqn. 19 
        retval = fc*( f*t0b + (1 - f)*t0ab ) ! Eqn. 17

        ! Transfer function: baryon part (Eqn. 21)
        st   = s / (1 + (beta_node/ks)**3)**(1._dp/3._dp)
        f    = sin(k*st) / (k*st) ! Spherical Bessel function, j0
        t2   = log( e + 1.8_dp*q )
        t0ab = t2 / (t2 + t3*q**2)
        t0b  = t0ab / (1 + (ks/5.2_dp)**2)     &
                + alpha_b / (1 + (beta_b/ks)**3) * exp(-(k/args%k_silk)**1.4_dp)
        retval = retval  + fb * t0b * f

        ! Interpolation using growth factor
        retval = dplus_z * retval 

        ! Linear matter power spectrum
        retval = (args%sigma8**2 * args%norm) * retval**2 * k**( args%ns )
        retval = log(retval)
        
    end function ps_eisenstein98_bao

end module power_spectrum_mod
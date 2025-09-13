module constants_mod
    
    use iso_c_binding
    implicit none
    
    real(c_double), parameter :: delta_sc = 1.6864701998411453_c_double
    !! Overdensity threshold for spherical collapse in EdS universe
    
    real(c_double), parameter :: critical_density_const = 2.775366e+07_c_double
    !! Constant part of the present critical density.
    !  -- 
    !  This value is calculated sung the relation for critical density at z=0, 
    !  `rho_c = 3*H0^2/(8*pi*G)`, where, G=6.6743e-11 m^3/kg/s^2 and H0 has 
    !  unit km/s/Mpc. Using 1 Mpc=3.085677e+22 m and Msun=1.9884099e+30 kg, 
    !  this value can be calculated (I used values in `astropy.constants`
    !  to calculate this value) 
    
    real(c_double), parameter :: pi = 3.141592653589793_c_double
    !! Pi

end module constants_mod
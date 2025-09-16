# Optimizing HOD parameters so that the calculated quantities match
# the observed values. 


import logging, numpy as np, numpy.typing as nt
from scipy.special import erf
from scipy.integrate import simpson
from scipy.optimize import minimize

def optimizer1(
        n_gal       : float,
        f_sat       : float,
        mass_func   : nt.NDArray[np.float64], 
        sigma_m     : float                 = 0.,
        alpha       : float                 = 1.,
        mmin_bounds : tuple[float, float]   = (1e+12, 1e+18),
        m1_bounds   : tuple[float, float]   = (1e+13, 1e+18),
        tol         : float                 = 1e-08,
        gridsize    : int | tuple[int, int] = (11, 11),
    ) -> tuple[float, float, float, float, float]:

    logger = logging.getLogger()

    sigma_m   = abs(sigma_m)
    mass_func = np.asarray(mass_func, dtype = 'f8')
    assert np.ndim(mass_func) == 2 and np.size(mass_func, 1) == 2

    # Getting the log of mass in Msun and dn/dln(m) (Mpc^-3) from the table:
    lnm, dndlnm = np.log(mass_func[:, 0]), mass_func[:, 1]

    def get_observables(p: tuple[float, float]) -> tuple[float, float]:
        lnm_min, lnm1 = p
        lnm0 = lnm_min    # setting M0 = M_min

        # Central galaxy count function
        arg       = lnm - lnm_min
        n_central = ( 
            0.5*(1. + erf(arg / sigma_m)) if sigma_m > 0. # sigmoid function
            else np.heaviside(arg, 1.)                    # step function, if sigma=0
        )
        
        # Satellite galaxy count function
        arg         = np.exp(lnm - lnm1) - np.exp(lnm0 - lnm1)
        n_satellite = n_central * np.maximum(arg, 0.)**alpha

        # Total galaxy density 
        ngal_calc = simpson( dndlnm * (n_central + n_satellite), lnm ) # Mpc^3
        
        # Satellite fraction
        fsat_calc = simpson( dndlnm * n_satellite, lnm ) / ngal_calc

        return ngal_calc, fsat_calc

    def cost(p: tuple[float, float], return_parts = False) -> float:
        # Cost is the weighted distance of the calculated values from the observed
        # values, with the observed values used as weights. 
        ngal_test, fsat_test = get_observables(p)
        cost = (ngal_test / n_gal - 1.)**2 + (fsat_test / f_sat - 1.)**2
        cost = np.log( 1e-08 + cost )
        return cost
    
    lnm_min_bounds = np.clip( np.log(mmin_bounds), min(lnm), max(lnm) )
    lnm1_bounds    = np.clip( np.log(m1_bounds)  , min(lnm), max(lnm) )
    logger.info(f"using actual log(m_min) bounds: {tuple(lnm_min_bounds)}")
    logger.info(f"using actual log(m1) bounds: {tuple(lnm1_bounds)}")

    # Guessing initial value using a grid:
    if isinstance(gridsize, int): gridsize = (gridsize, )*2

    def gridp(i: int, j: int) -> tuple[float, float]: 
        # Interpolate the values of log(M_min) and log(M1) on the grid.
        return (
            np.interp( i, [0, gridsize[0]-1], lnm_min_bounds ), 
            np.interp( j, [0, gridsize[1]-1], lnm1_bounds    ), 
        )
    
    initial_guess = gridp(
        *np.unravel_index( 
            np.argmin(
                np.fromfunction( 
                    np.vectorize(lambda i, j: cost( gridp(i, j) ), otypes=[float]), 
                    gridsize,
                )
            ),
            gridsize, 
        )
    )
    logger.info(f"using initial guess: {initial_guess}")

    # Optimization:
    optim_result = minimize(
        cost, 
        x0     = initial_guess,
        bounds = [ lnm_min_bounds, lnm1_bounds ], 
        tol    = tol,
    )
    if optim_result.success:
        logger.info(f"optimization successful, message: {optim_result.message!r}")
        logger.info(f"reached optimum value: {optim_result.x} in {optim_result.nit} iterations")
    else:
        logger.info(f"optimization failed, message: {optim_result.message!r}")
    lnm_min, lnm1 = float(optim_result.x[0]), float(optim_result.x[1])

    ngal_opt, fsat_opt = get_observables(optim_result.x)
    logger.info(f"calculated values: n_gal={ngal_opt:.6g}, f_sat={fsat_opt:.6g}")
    
    # Optimum HOD parameters: 
    #               log(M_min) sigma_M  log(M0)  log(M1) alpha
    optimum_hod = ( lnm_min,   sigma_m, lnm_min, lnm1,   alpha )
    return optimum_hod

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     mass_func = np.exp( np.loadtxt('_data/huge.hmf.dat') )
#     p = optimizer1(
#         2.5e-05, 
#         0.05, 
#         mass_func, 
#         0., 
#         1., 
#     )
#     print(p)

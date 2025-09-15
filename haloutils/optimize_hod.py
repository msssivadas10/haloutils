# Optimizing HOD parameters so that the calculated quantities match
# the observed values. 


import logging, numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from scipy.optimize import minimize

def optimizer1(
        n_gal       : float,
        f_sat       : float,
        mass_func   : list[tuple[float, float]], 
        sigma_m     : float                 = 0.,
        alpha       : float                 = 1.,
        mmin_bounds : tuple[float, float]   = (1e+12, 1e+18),
        m1_bounds   : tuple[float, float]   = (1e+13, 1e+18),
        tol         : float                 = 1e-08,
        gridsize    : int | tuple[int, int] = (11, 11)
    ) -> tuple[float, float, float, float, float]:

    logger = logging.getLogger()

    sigma_m   = abs(sigma_m)
    mass_func = np.asarray(mass_func, dtype = 'f8')
    assert np.ndim(mass_func) == 2 and np.size(mass_func, 1) == 2

    # Getting the log of mass in Msun and dn/dln(m) (Mpc^-3) from the table:
    lnm, dndlnm = np.log(mass_func[:, 0]), mass_func[:, 1]

    def cost(p: tuple[float, float], return_parts = False) -> float:

        lnm_min, lnm1 = p
        lnm0 = lnm_min 

        # Central galaxy count function
        arg = lnm - lnm_min
        if sigma_m < 1e-08: 
            central_count = np.heaviside(arg, 1.)
        else:
            central_count = 0.5*( 1. + erf(arg / sigma_m) )
        
        # Satellite galaxy count function
        arg = np.exp(lnm - lnm1) - np.exp(lnm0 - lnm1)
        satellite_count = central_count * np.maximum(arg, 0.)**alpha

        # Total galaxy density 
        ngal_test = simpson( dndlnm * (central_count + satellite_count), lnm ) # Mpc^3
        # return ngal_test / n_gal
        
        # Satellite fraction
        fsat_test = simpson( dndlnm * satellite_count, lnm ) / ngal_test
        # return fsat_test / f_sat

        # Cost is the weighted distance of the calculated values from the observed
        # values, with the observed values used as weights. 
        cost = (ngal_test / n_gal - 1.)**2 + (fsat_test / f_sat - 1.)**2
        cost = np.log( 1e-08 + cost )

        return cost if not return_parts else ( cost, (ngal_test, fsat_test) )
    
    lnm_min_bounds = np.clip( np.log(mmin_bounds), min(lnm), max(lnm) )
    lnm1_bounds    = np.clip( np.log(m1_bounds)  , min(lnm), max(lnm) )
    logger.info(f"using actual log(m_min) bounds: {tuple(lnm_min_bounds)}")
    logger.info(f"using actual log(m1) bounds: {tuple(lnm1_bounds)}")

    # Guessing initial value using a grid:
    if isinstance(gridsize, int): gridsize = (gridsize, )*2
    lnm_min_grid = np.linspace(*lnm_min_bounds, gridsize[0])
    lnm1_grid    = np.linspace(*lnm1_bounds   , gridsize[1])
    guess_index  = np.unravel_index( 
        np.argmin([[ cost([a, b]) for b in lnm1_grid ] for a in lnm_min_grid ]), 
        lnm_min_grid.shape+lnm1_grid.shape, 
    )
    initial_guess = ( lnm_min_grid[guess_index[0]], lnm1_grid[guess_index[1]] )
    logger.info(f"using initial guess: {tuple(initial_guess)}")

    # Optimization:
    optim_result = minimize(
        cost, 
        x0     = initial_guess,
        bounds = [ lnm_min_bounds, lnm1_bounds ], 
        tol    = tol,
    )
    if optim_result.success:
        logger.info(f"optimization successful, message: {optim_result.message!r}")
        logger.info(f"reached optimum value: {tuple(optim_result.x)} in {optim_result.nit} iterations")
    else:
        logger.info(f"optimization failed, message: {optim_result.message!r}")

    ngal_opt, fsat_opt = cost(optim_result.x, return_parts = True)[1]
    logger.info(f"calculated values: n_gal={ngal_opt:.6g}, f_sat={fsat_opt:.6g}")
    
    optimum_hod = ( optim_result.x[0], sigma_m, optim_result.x[0], optim_result.x[1], alpha )
    return optimum_hod

logging.basicConfig(level=logging.INFO)
mass_func = np.exp( np.loadtxt('x.dat') )
p = optimizer1(
    2.5e-05, 
    0.05, 
    mass_func, 
    0., 
    1., 
)
print(p)

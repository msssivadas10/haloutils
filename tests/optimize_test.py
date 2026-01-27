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
        beta        : float                 =-1.,
        mmin_bounds : tuple[float, float]   = (1e+12, 1e+14),
        m1_bounds   : tuple[float, float]   = (1e+13, 1e+14),
        tol         : float                 = 1e-08,
        gridsize    : int | tuple[int, int] = (11, 11),
        save_to     : str                   = '',
    ) -> tuple[float, float, float, float, float]:

    def calculate_lnm0(lnm_min, sigma_m = 0., beta = -1.):
        # Calculate the value of M0 based on Mmin and sigma_m values. Also add
        # the offset corresponding parameter beta, so that no satellites are 
        # created for Mmin < beta*M...  
        lnm0 = lnm_min - 3*abs(sigma_m)
        if beta > 0.: 
            lnm0 = lnm0 + np.log(beta)
        return lnm0

    logger = logging.getLogger()

    sigma_m    = abs(sigma_m)
    mass_func  = np.asarray(mass_func, dtype = 'f8', copy=True)
    assert np.ndim(mass_func) == 2 and np.size(mass_func, 1) == 2

    # Getting the log of mass in Msun and dn/dln(m) (Mpc^-3) from the table:
    lnm, dndlnm = mass_func[:, 0], np.exp( mass_func[:, 1] )

    def get_observables(p: tuple[float, float]) -> tuple[float, float]:
        lnm_min, lnm1 = p
        lnm0          = calculate_lnm0(lnm_min, sigma_m, beta)

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
    
    data = np.fromfunction( 
        np.vectorize(lambda i, j: cost( gridp(i, j) ), otypes=[float]), 
        gridsize,
    )
    initial_guess = gridp(
        *np.unravel_index( 
            np.argmin(
                data
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
    lnm0          = calculate_lnm0(lnm_min, sigma_m, beta)
    
    ngal_opt, fsat_opt = get_observables(optim_result.x)
    logger.info(f"calculated values: n_gal={ngal_opt:.6g}, f_sat={fsat_opt:.6g}")
    
    # Optimum HOD parameters: 
    #               log(M_min) sigma_M  log(M0)  log(M1) alpha
    optimum_hod = ( lnm_min,   sigma_m, lnm0,    lnm1,   alpha )

    import matplotlib.pyplot as plt
    plt.contourf(
        np.interp( np.arange(gridsize[0]), [0, gridsize[0]-1], lnm_min_bounds ), 
        np.interp( np.arange(gridsize[1]), [0, gridsize[1]-1], lnm1_bounds    ),
        data.T,
        levels=21,
    )
    plt.plot( [ initial_guess[0]], [ initial_guess[1]], 'xk' )
    plt.plot( [optim_result.x[0]], [optim_result.x[1]], '+k' )
    plt.colorbar()
    plt.xlabel("log m_min"); plt.ylabel("log m_1")
    plt.show()

    if save_to and isinstance(save_to, str):
        # Save results to a file:
        import asdf
        with asdf.AsdfFile({
            "lnm_min" : lnm_min,   
            "sigma_m" : sigma_m, 
            "lnm0"    : lnm0,    
            "lnm1"    : lnm1,   
            "alpha"   : alpha,
            "ngal_opt": ngal_opt, 
            "fsat_opt": fsat_opt,
            "options" : {
                "n_gal"      : n_gal,
                "f_sat"      : f_sat,
                "sigma_m"    : sigma_m,
                "alpha"      : alpha,
                "beta"       : beta,
                "mmin_bounds": mmin_bounds,
                "m1_bounds"  : m1_bounds,
                "tol"        : tol,
                "gridsize"   : gridsize,
            }, 
            "data": {
                "mass_func"  : mass_func,
                "cost_grid"  : data,
            }, 
            "optimization_result": {
                "initial" : tuple([ *initial_guess  ]),
                "optimum" : tuple([ *optim_result.x ]),
                "message" : optim_result.message,
                "success" : optim_result.success,
                "nit"     : optim_result.nit,
            }
        }) as af: af.write_to(save_to)
    
    return optimum_hod

import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

mass_func1      = np.loadtxt('_data/hmf.txt') 
mass_func2      = np.empty_like(mass_func1)
mass_func2[:,0] = np.linspace(np.log(1e+08), np.log(1e+15), mass_func2.shape[0])
mass_func2[:,1] = np.polyval(
    np.polyfit( mass_func1[:,0], mass_func1[:,1], 2 ), 
    mass_func2[:,0]
)
mask = mass_func2[:,0] < mass_func1[0,0]
mass_func2[mask,1] = np.polyval(
    np.polyfit( mass_func2[~mask,0][:2], mass_func2[~mask,1][:2], 1 ), 
    mass_func2[mask,0]
)
mask = mass_func2[:,0] > mass_func1[-1,0]
mass_func2[mask,1] = np.polyval(
    np.polyfit( mass_func2[~mask,0][-2:], mass_func2[~mask,1][-2:], 1 ), 
    mass_func2[mask,0]
) 
# plt.plot(*mass_func1.T)
# plt.plot(*mass_func2.T)
# plt.show()

p = optimizer1(
    2.5e-04, 
    0.2, 
    mass_func2, 
    0., 
    1., 
    beta=2.,
    mmin_bounds=(1e+10, 1e+14),
    m1_bounds=(1e+10, 1e+14),
    gridsize=(51, 55)
)
print(np.exp([p[0], p[2], p[3]]))

# lnm_min, sigma_m, lnm0, lnm1, alpha = p

# m = np.logspace(12, 14, 21)
# lnm = np.log(m)
# arg       = lnm - lnm_min
# n_central = ( 
#     0.5*(1. + erf(arg / sigma_m)) if sigma_m > 0. 
#     else np.heaviside(arg, 1.)                    
# )
# arg         = ( np.exp(lnm) - np.exp(lnm0) ) / np.exp(lnm1)
# n_satellite = n_central * np.maximum(arg, 0.)**alpha

# plt.loglog()
# plt.plot(m, n_central  , 's-')
# plt.plot(m, n_satellite, 's-')
# plt.show()
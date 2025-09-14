import numpy as np, math, time

try:
    # Works only if the module is available in path
    from haloutils.galaxy_catalog_generator import _share_data, _generate_galaxies, _clean_up
    from haloutils import Eisenstein98_zb, HaloModel, halo_massfunction

except ModuleNotFoundError:
    # Loading module from file:
    import sys, os.path
    from importlib.util import spec_from_file_location, module_from_spec

    __file_path = os.path.abspath( os.path.dirname(os.path.dirname(__file__)) )
    module_name = 'haloutils'
    module_path = os.path.join(__file_path, module_name, '__init__.py')
    spec        = spec_from_file_location(module_name, module_path) # Create a module specification 
    haloutils   = module_from_spec(spec) # Create a new module object based on the specification
    sys.modules[module_name] = haloutils # Add the module to sys.modules 
    spec.loader.exec_module(haloutils)   # Execute the module's code

    from haloutils.galaxy_catalog_generator import _share_data, _generate_galaxies, _clean_up
    from haloutils import Eisenstein98_zb, HaloModel, halo_massfunction

m_max = 1e+14

halo_model = HaloModel(
    Eisenstein98_zb(
        z      = 0.  , H0     = 70.0, 
        Om0    = 0.3 , Ob0    = 0.05, 
        Ode0   = 0.7 , ns     = 1.  , 
        sigma8 = 1.  , 
    ), 
    lnm_min    = np.log(2.563e+12), sigma_m    = 0.,
    lnm0       = np.log(2.563e+12), lnm1       = np.log(6.147e+13),
    alpha      = 1.               , scale_shmf = 0.5      ,
    slope_shmf = 2.               , Delta_m    = 200.     ,
)

def calculate_mass_function_dndlnm():
    # Calculate halo mass-function table.

    lnm_vals = np.linspace( halo_model.lnm_min, math.log(m_max), 101 )
    hmf_vals = halo_massfunction( 
        lnm_vals, 
        'tinker08', 
        ps      = halo_model.ps, 
        Delta_m = halo_model.Delta_m, 
        filt    = 'tophat',
        retval  = 'dn/dlnm', 
    )[0]
    return lnm_vals, hmf_vals

def generate_halos(m_vals, hmf_vals, n_samples, Lbox, rng=None):
    # Generate random halo catalog given tabulated (m, dn/dlnm).
    
    if rng is None:
        rng = np.random.default_rng()
    
    m_vals   = np.asarray(m_vals)
    hmf_vals = np.asarray(hmf_vals)

    # compute PDF in m: p(m) ∝ hmf(m) / m
    pdf = hmf_vals / m_vals
    cdf = np.cumsum(pdf * np.gradient(m_vals))
    cdf /= cdf[-1]  # normalize to [0,1]
    
    # inverse transform sampling
    u = rng.random(n_samples)
    masses = np.interp(u, cdf, m_vals)

    # position
    positions = rng.uniform(0., Lbox, size=[n_samples, 3])

    # unique ids
    ids = np.arange(1, n_samples+1)
    return ids, positions, masses

def generate_galaxies(Lbox, ids, positions, masses, rseed=None):
    # Generate galaxy catalog using halo catalog.

    _share_data(
        0, 
        {
            "lnm_min"   : halo_model.lnm_min, 
            "sigma_m"   : halo_model.sigma_m, 
            "lnm0"      : halo_model.lnm0, 
            "lnm1"      : halo_model.lnm1, 
            "alpha"     : halo_model.alpha, 
            "scale_shmf": halo_model.scale_shmf, 
            "slope_shmf": halo_model.slope_shmf, 
            "z"         : halo_model.z,
            "H0"        : halo_model.H0, 
            "Om0"       : halo_model.Om0, 
            "Delta_m"   : halo_model.Delta_m, 
            "dplus"     : halo_model.dplus,
        }, 
        [[0.]*3,[Lbox]*3], 
        [halo_model.lnm_min, math.log(m_max), 101, 0], 
        halo_model.ps._table, 
        [(ids, positions, masses)]
    )
    _generate_galaxies(0, 8, rseed=rseed or int(time.time()))
    with open('0.gbuf.dat', 'r') as fp:
        galaxies = np.fromfile(
            fp, 
            dtype = [("id", "<i8"), ("pos", "<f8", 3), ("mass", "<f8"), ("typ", "S1")], 
        )
    _clean_up(0)

    # Galaxy counts:
    count_dict = dict(zip(*np.unique(galaxies['typ'], return_counts=True)))
    N_c, N_s   = count_dict[b'c'], count_dict[b's']

    return galaxies, (N_c, N_s)

def check_counts():

    Lbox = 500.0/halo_model.ps.h  # Mpc
    V    = Lbox**3

    lnm_vals, hmf_vals = calculate_mass_function_dndlnm()
    m_vals = np.exp(lnm_vals)

    # Calculating halo density: 
    halo_density = np.trapz(hmf_vals, lnm_vals) # Mpc^3
    N_halos_exp  = int(V * halo_density)

    # Catalog generation:
    rng = np.random.default_rng(42)
    ids, positions, masses = generate_halos(m_vals, hmf_vals, N_halos_exp, Lbox, rng)
    galaxies, (N_c, N_s)   = generate_galaxies(Lbox, ids, positions, masses, rseed=None)

    # Empirical:
    N_g     = N_c + N_s # total galaxy count
    n_g_emp = N_g / V   # total galaxy density Mpc^3
    f_s_emp = N_s / N_g # satellite fraction
    # Theory:
    n_g_theory = np.trapz( 
        hmf_vals*halo_model.central_count(lnm_vals)*(1. + halo_model.satellite_count(lnm_vals)), 
        lnm_vals, 
    ) # total galaxy density Mpc^3
    f_s_theory = np.trapz( hmf_vals*halo_model.satellite_count(lnm_vals), lnm_vals) / n_g_theory # satellite fraction
    N_g_theory = n_g_theory * V # total galaxy count

    print(f"Halos    : {N_halos_exp}")
    print(f"Theory   : n_g = {n_g_theory:.3e}, f_s = {f_s_theory:.3f}, N_g = {N_g_theory:.1f}")
    print(f"Empirical: n_g = {n_g_emp:.3e}, f_s = {f_s_emp:.3f}, N_g = {N_g}")



check_counts()

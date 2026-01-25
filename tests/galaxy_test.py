import numpy as np, math, time
import matplotlib.pyplot as plt

try:
    # Works only if the module is available in path
    from haloutils.galaxy_catalog_generator import pack_arguments, generate_galaxies
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

    from haloutils.galaxy_catalog_generator import pack_arguments, generate_galaxies
    from haloutils import Eisenstein98_zb, HaloModel, halo_massfunction

m_min = 1e+12
m_max = 1e+14

halo_model = HaloModel(
    Eisenstein98_zb(
        z      = 0.  , H0     = 70.0, 
        Om0    = 0.3 , Ob0    = 0.05, 
        Ode0   = 0.7 , ns     = 1.  , 
        sigma8 = 1.  , 
    ), 
    lnm_min    = np.log(1e+12), sigma_m    = 0.,
    lnm0       = np.log(1e+12), lnm1       = np.log(1e+13),
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

def generate_galaxies_(Lbox, ids, positions, masses, rseed=None):
    # Generate galaxy catalog using halo catalog.

    args = pack_arguments(
        "_data/work", 
        (
            halo_model.lnm_min, 
            halo_model.sigma_m, 
            halo_model.lnm0, 
            halo_model.lnm1, 
            halo_model.alpha, 
            halo_model.scale_shmf, 
            halo_model.slope_shmf, 
            halo_model.z,
            halo_model.H0, 
            halo_model.Om0, 
            halo_model.Delta_m, 
            halo_model.dplus,
        ), 
        [[0.]*3,[Lbox]*3], 
        np.log([ m_min, m_max ]),
        101, 
        0,
        halo_model.ps._table,
        rseed, 
        4,
        [(ids, positions, masses)],
    )

    generate_galaxies(args)
    with open("_data/work/gbuf.bin", 'r') as fp:
        galaxies = np.fromfile(
            fp, 
            dtype = [("id", "<i8"), ("pos", "<f8", 3), ("mass", "<f8"), ("typ", "S1")], 
        )
    # _clean_up(0)

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
    halo_density = np.trapezoid(hmf_vals, lnm_vals) # Mpc^3
    N_halos_exp  = int(V * halo_density)

    # Catalog generation:
    rng = np.random.default_rng(42)
    ids, positions, masses = generate_halos(m_vals, hmf_vals, N_halos_exp, Lbox, rng)
    galaxies, (N_c, N_s)   = generate_galaxies_(Lbox, ids, positions, masses, rseed=None)

    # with open("_data/work/hbuf.bin", 'r') as fp:
    #     halos = np.fromfile(
    #         fp, 
    #         dtype = [("id", "<i8"), ("pos", "<f8", 3), ("mass", "<f8")], 
    #     )
    # dndlnm, lnm_bins = np.histogram( halos["mass"], np.logspace(np.log10(m_min), np.log10(m_max), 15) )
    # lnm_bins = np.log(lnm_bins)
    # dndlnm   = dndlnm / np.diff(lnm_bins) / Lbox**3
    # lnm_bins = ( lnm_bins[1:] + lnm_bins[:-1] ) / 2.
    # plt.plot(lnm_bins, np.log(dndlnm), 's')
    # plt.plot(lnm_vals, np.log(hmf_vals))
    # plt.show()

    # Empirical:
    N_g     = N_c + N_s # total galaxy count
    n_g_emp = N_g / V   # total galaxy density Mpc^3
    f_s_emp = N_s / N_g # satellite fraction
    # Theory:
    n_g_theory = np.trapezoid( 
        hmf_vals*halo_model.central_count(lnm_vals)*(1. + halo_model.satellite_count(lnm_vals)), 
        lnm_vals, 
    ) # total galaxy density Mpc^3
    f_s_theory = np.trapezoid( hmf_vals*halo_model.satellite_count(lnm_vals), lnm_vals) / n_g_theory # satellite fraction
    N_g_theory = n_g_theory * V # total galaxy count

    print(f"Halos    : {N_halos_exp}")
    print(f"Theory   : n_g = {n_g_theory:.3e}, f_s = {f_s_theory:.3f}, N_g = {N_g_theory:.1f}")
    print(f"Empirical: n_g = {n_g_emp:.3e}, f_s = {f_s_emp:.3f}, N_g = {N_g}")

def check_counts2():

    Lbox = 500.0/halo_model.ps.h  # Mpc
    V    = Lbox**3
    m_halo = 8e+12
    lnm_halo = np.log(m_halo)

    lnm_vals, hmf_vals = calculate_mass_function_dndlnm()
    m_vals = np.exp(lnm_vals)

    # Calculating halo density: 
    N_halos_exp  = 100000

    # Catalog generation:
    rng = np.random.default_rng()#(42)
    ids = np.arange(N_halos_exp)
    positions = np.zeros((N_halos_exp, 3), dtype="f8")
    masses    = np.ones((N_halos_exp, ), dtype="f8") * m_halo 
    galaxies, (N_c, N_s)   = generate_galaxies_(Lbox, ids, positions, masses, rseed=None)

    mask = galaxies["typ"] == b's'
    data = galaxies["pos"][mask]
    r_gal = np.sqrt(np.sum(np.square(data), axis=1))
    x_gal = galaxies["mass"][mask]

    # r_vir = np.exp( halo_model.lagrangian_r(lnm_halo) )
    # c     = halo_model.halo_concentration(lnm_halo)
    # from scipy.interpolate import CubicSpline
    # ca = np.logspace(-3, 1.5, 101)
    # aa = np.log(1+ca) - ca/(1+ca)
    # f  = CubicSpline(aa, ca)
    # x  = np.random.uniform(0., 1., size=N_halos_exp) * np.log(1+c) - c/(1+c)
    # x  = f(x)
    # r_gal = x*r_vir/c

    r_vir = np.exp( halo_model.lagrangian_r(lnm_halo) )
    c     = halo_model.halo_concentration(lnm_halo)
    rs    = r_vir / c
    bins  = np.logspace(np.log10(1e-3*r_vir), np.log10(r_vir), 15)
    hist, edges = np.histogram(r_gal, bins=bins, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    rho_theory = 1/(centers/rs*(1+centers/rs)**2)
    rho_theory /= np.trapezoid(rho_theory*centers**2, centers) # normalize
    plt.loglog(centers, hist, 's--', label='Sampled')
    plt.loglog(centers, rho_theory*centers**2, label='Theory')
    plt.xlabel("r")
    plt.ylabel("$r^2 \\rho(r)$")
    plt.show()

    # bins   = np.logspace(np.log10(1e-3*m_halo), np.log10(m_halo), 50)
    # hist, edges = np.histogram(x_gal, bins=bins, density=True)
    # centers = 0.5*(edges[1:] + edges[:-1])
    # rho_theory = halo_model.subhalo_massfunction(centers/m_halo, np.log(m_halo))
    # rho_theory /= np.trapezoid(rho_theory, centers) # normalize
    # plt.loglog(centers, hist, 's--', label='Sampled')
    # plt.loglog(centers, rho_theory, label='Theory')
    # plt.xlabel("$m_sat$")
    # plt.ylabel("SHMF$")
    # plt.show()

    
# check_counts()
check_counts2()


# def f(x): return np.log(1 + x) - x / (1 + x)
# def f2(x): return np.log(1 + x) - 1. # x > 50
# def f2(x): return x**2/2 # x < 1e-3

# c = np.logspace(-6, 3, 201)
# y = f(c)
# def f2(x): return np.interp(x, c, y)

# a = 15

# y = [ a ]
# for i in range(10):
#     x = y[-1]
#     x = x - ( np.log(1 + x) - x / (1 + x) - a ) * ( (1 + x)**2 / x )
#     y.append(x)

# print( y[-1], f(y[-1]) )
# print( f(np.exp(1+a)-1), np.exp(1+a)-1 )

# plt.plot(y, 's-')
# plt.show()

# t = log(1+x) - 1
# t+1 = log(1+x)
# exp(t+1) = 1+x
# x = exp(1+t) - 1 
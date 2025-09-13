
import numpy as np
import matplotlib.pyplot as plt

try:
    # Works only if the module is available in path
    from haloutils import Eisenstein98_zb, HaloModel, halo_massfunction, halo_bias

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

    from haloutils import Eisenstein98_zb, HaloModel, halo_massfunction, halo_bias
    
    
ps = Eisenstein98_zb(z = 0., H0 = 70., Om0 = 0.3, Ob0 = 0.05, Ode0 = 0.7, ns = 1., sigma8 = 1.)
hm = HaloModel(
    ps, 
    lnm_min    = np.log(2.563e+12),
    sigma_m    = 0.,
    lnm0       = np.log(2.563e+12),
    lnm1       = np.log(6.147e+13),
    alpha      = 1.  ,
    scale_shmf = 0.5 ,
    slope_shmf = 2.  ,
    Delta_m    = 200.,
)

def test_linear_growth():
    z = np.linspace(0., 10, 11)
    y = ps.linear_growth(z, 0)
    plt.plot(z, y, 'o-')
    plt.xlabel('$z$')
    plt.ylabel('$D_+(z)$')
    plt.show()

def test_power_spectrum():
    k = np.logspace(-4, 3, 11)
    y = ps.power( np.log(k) )
    plt.loglog(k, y, '-o')
    plt.xlabel('k in 1/Mpc')
    plt.ylabel('P(k, z) in Mpc^3')
    plt.show()

def test_variance():
    r = np.logspace(-3, 2, 11)
    s = ps.spectral_moment( np.log(r), 0, 0, 'tophat' )
    plt.loglog(r, s, '-o')
    plt.xlabel('r in Mpc')
    plt.ylabel('$\\sigma^2$(r, z)')
    plt.show()

def test_correlation():
    r = np.logspace(-3, 2, 11)
    c = ps.correlation( np.log(r), 0 )
    plt.loglog(r, c, '-o')
    plt.xlabel('r in Mpc')
    plt.ylabel('corr(r, z)')
    plt.show()

def test_mass_function_models():
    import matplotlib.cm
    all_models = ['press74', 'sheth01', 'jenkins01', 'reed03', 'tinker08', 'courtin10', 'crocce10']
    all_colors = matplotlib.cm.get_cmap('viridis')( np.arange(len(all_models)) / (len(all_models) - 1) )

    m = np.logspace(6, 16, 21)
    f = np.zeros_like(m)
    for c, model in zip(all_colors, all_models):
        f, s, _ = halo_massfunction( np.log(m), model, ps, 200., retval = 'fs' )
        plt.semilogx(s, f, 'o-', color = c, label = model.title())
    plt.legend(title = 'halo massfunction')
    plt.xlabel('s')
    plt.ylabel('$f(\\sigma)$')
    plt.show()

def test_mass_function_tinker08():
    m = np.logspace(8, 16, 11)
    n, *_ = halo_massfunction( np.log(m), 'tinker08', ps, 200., retval = 'dn/dm' )
    plt.loglog(m, n, 'o-')
    plt.xlabel('$m_{halo}$ in Msun')
    plt.ylabel('$\\frac{dn}{dm}(m_{halo}, z)$ in 1/Mpc^3/Msun')
    plt.show()

def test_bias_models():
    all_models = ['cole89', 'sheth01', 'tinker10']
    all_colors = ['tab:blue', 'green', 'orange']

    m = np.logspace(6, 16, 21)
    f = np.zeros_like(m)
    for c, model in zip(all_colors, all_models):
        f, s, _ = halo_bias( np.log(m), model, ps, 200. )
        plt.loglog(1.686/s, f, 'o-', color = c, label = model.title())
    plt.legend(title = 'halo bias')
    plt.xlabel('$m_{halo}$ in Msun')
    plt.ylabel('halo bias $b(m_{halo}, z)$')
    plt.show()

def test_galaxy_count():
    m = np.logspace(8, 14, 21)
    nc = hm.central_count(np.log(m))
    ns = hm.satellite_count(np.log(m))
    plt.semilogx(m, nc, 'o-', label="central")
    plt.semilogx(m, ns, 'o-', label="satellite")
    plt.legend(title = 'galaxy')
    plt.xlabel('$m_{halo}$ in Msun')
    plt.ylabel('galaxy count')
    plt.show()

def test_shmf():
    x = np.linspace(0., hm.scale_shmf+0.1, 21)
    y = hm.subhalo_massfunction(x, np.log(1e+13))
    plt.plot(x, y, 'o-')
    plt.xlabel('$m_{sat}/m_{halo}$ in Msun')
    plt.ylabel('SHMF')
    plt.show()

def test_halo_concentration():
    m = np.logspace(8, 16, 21)
    c = hm.halo_concentration(np.log(m))
    plt.semilogx(m, c, 'o-')
    plt.xlabel('$m_{halo}$ in Msun')
    plt.ylabel('$c(m_{halo}, z)$')
    plt.show()

def test_galaxy_generation():
    lnm   = np.log(1e+14)
    data  = hm.generate_galaxies(lnm, count = 200_000)
    r_gal = np.sqrt(np.sum(np.square(data[1:,0:3]), axis=1))
    x_gal = data[1:,3]

    r_vir = np.exp( hm.lagrangian_r(lnm) )
    c     = hm.halo_concentration(lnm)
    rs    = r_vir / c
    bins  = np.logspace(np.log10(1e-3*r_vir), np.log10(r_vir), 50)
    hist, edges = np.histogram(r_gal, bins=bins, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    rho_theory = 1/(centers/rs*(1+centers/rs)**2)
    rho_theory /= np.trapz(rho_theory*centers**2, centers) # normalize
    plt.loglog(centers, hist, label='Sampled')
    plt.loglog(centers, rho_theory*centers**2, label='Theory')
    plt.xlabel("r")
    plt.ylabel("$r^2 \\rho(r)$")
    plt.show()

    m_halo = np.exp(lnm)
    bins   = np.logspace(np.log10(1e-3*m_halo), np.log10(m_halo), 50)
    hist, edges = np.histogram(x_gal, bins=bins, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    rho_theory = hm.subhalo_massfunction(centers/m_halo, np.log(m_halo))
    rho_theory /= np.trapz(rho_theory, centers) # normalize
    plt.loglog(centers, hist, label='Sampled')
    plt.loglog(centers, rho_theory, label='Theory')
    plt.xlabel("$m_sat$")
    plt.ylabel("SHMF$")
    plt.show()



test_linear_growth()
test_power_spectrum()
test_variance()
test_correlation()
test_mass_function_models()
test_mass_function_tinker08()
test_bias_models()
test_galaxy_count()
test_shmf()
test_halo_concentration()
test_galaxy_generation()

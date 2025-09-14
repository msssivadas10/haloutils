import numpy as np
import matplotlib.pyplot as plt

try:
    # Works only if the module is available in path
    from haloutils.correlation import Grid

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

    from haloutils.correlation import Grid

def pair_count(points1, points2, bin_edges, boxsize=None, autocorr=False, use_r2=False):
    # Pair counting refernce: brute force method.

    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)
    counts = np.zeros(len(bin_edges) - 1, dtype=int)

    if np.ndim(boxsize) == 0 or boxsize is None:
        box = None
    else:
        box = np.array(boxsize, dtype=float)

    N1 = len(points1)
    N2 = len(points2)

    if autocorr:
        # same set: only loop i < j
        for i in range(N1):
            for j in range(i + 1, N1):
                d = points1[j] - points1[i]
                if box is not None:
                    d -= box * np.round(d / box)

                dist2 = np.dot(d, d)
                val = dist2 if use_r2 else np.sqrt(dist2)

                k = np.searchsorted(bin_edges, val, side='right') - 1
                if 0 <= k < len(counts):
                    counts[k] += 1
    else:
        # two different sets: full cross product
        for i in range(N1):
            for j in range(N2):
                d = points2[j] - points1[i]
                if box is not None:
                    d -= box * np.round(d / box)

                dist2 = np.dot(d, d)
                val = dist2 if use_r2 else np.sqrt(dist2)

                k = np.searchsorted(bin_edges, val, side='right') - 1
                if 0 <= k < len(counts):
                    counts[k] += 1

    return counts

def get_timings():

    from time import perf_counter

    def time_pair_counting(npts):

        boxsize, origin = [10., 10., 10.], [0., 0., 0.] 
        rng      = np.random.default_rng()
        pos      = rng.uniform(origin, boxsize, (npts, 3))
        rbins    = np.linspace(0., 4., 5)
        autocorr = False 
        periodic = False

        t_start  = perf_counter()
        grid     = Grid(pos, boxsize=boxsize, gridsize=[2, 2, 2], origin=origin)
        cnts     = grid.count_neighbours(grid, rbins, autocorr=autocorr, periodic=periodic)
        t_total  = perf_counter() - t_start
        # print("pair counts: ", cnts)

        t_start  = perf_counter() 
        cnts2    = pair_count(pos, pos, rbins, grid.boxsize if periodic else None, autocorr)
        t_total2 = perf_counter() - t_start
        # print("pair counts: ", cnts2)

        assert np.allclose(cnts, cnts2)
        t_total, t_total2 = 1000*t_total, 1000*t_total2

        print(f"npts: {npts}")
        print(f" - grid based pair counting: {t_total:3f} ms")
        print(f" - explicit pair counting  : {t_total2:.3f} ms")
        return t_total, t_total2


    grid_timings = {}
    bfpc_timings = {}
    for npts in [10, 20, 50, 100, 500, 1000, 2000]:
        grid_timings[npts], bfpc_timings[npts] = time_pair_counting(npts)
        
    plt.loglog()
    plt.plot(grid_timings.keys(), grid_timings.values(), 'o-', label="grid-based")
    plt.plot(bfpc_timings.keys(), bfpc_timings.values(), 'o-', label="brute-force")
    plt.legend()
    plt.show()


get_timings()
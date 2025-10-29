# Calculating halo mass-function from simulations.
# 
# Usage: 
#   python haloutils.mass_function COMMAND [OPTIONS]
# 
# Commands:
# - abacus-massfunction : for abacus summit halo catalogs
#
# Note: To pass bin edges, loaded from a file or calculated with Python, 
# use the BINS environment variable, instead of option --bins. For this,
# first the space-seperated values are assigned to the BINS variable, 
# export it and then run the script.
#
# To use values calculated with python, 
# 
#   BINS=$(python -c "import numpy as np; print(np.logspace(12, 15, 101))") # generate values
#   BINS=$(echo $BINS | tr '\n' ' ' | grep -oP '(?<=\[)[^]]*(?=\])')        # get values between []
#  

import os, re, logging, multiprocessing, numpy as np
from pathlib import Path
from itertools import repeat
from functools import reduce  

__version__ = "0.1a"

# A wrapper around `numpy.histogram`, that only return the counts.
def _histogram(*args, **kwargs): return np.histogram(*args, **kwargs)[0]

def abacus_massfunction(
        simname  : str, 
        redshift : float, 
        bins     : list[float],
        outfile  : str = None,
        loc      : str = '.', 
        nprocs   : int = -1,
        smooth   : int =  0,
    ):
    """
    Calculate halo mass-function from abacus summit halo catalog.\f

    Parameters
    ----------
    simname : str
        Name of the simulation.

    redshift : float
        Redshift value. 

    bins : array_like
        Mass bin edges in Msun unit. 

    outfile : str, optional
        If given, write the data to this path as a table of `log(mass)` and `log(dndlnm)`.
        `mass` and `dndlnm` have units Msun and Mpc^-3 respecively. 

    loc : str, optional
        Path to look for catalog files. If no files are found, raise `FileNotFoundError`.

    nprocs : int, optional
        Number of parallel threads to use for calculations.

    smooth : int, default=0
        If given a value > 1, smooth the data using a Boxcar filter of that size.

    Returns
    -------
    centers : array_like
        Mass corresponding to bin centers, in Msun. Bins containing no halos will be 
        removed. So, `len(centers) <= len(bins)-1` in general.

    dndlnm : array_like
        Halo mass-function `dn/dln(m)` in Mpc^-3 - number density of halos in unit log
        mass interval. Same size as `centers`.

    """
    import asdf
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    logger = logging.getLogger()

    # Filter out only the files with names matching the pattern and sorting based on 
    # the integer index: 
    files = [ 
        m.string 
            for m in map( 
                lambda fn: re.search(r"halo_info_(\d{3}).asdf", str(fn)), # for files like `halo_info_123.asdf`
                Path(loc).expanduser().absolute()                         # full path to files location
                         .joinpath(simname, "halos", f"z{redshift:.3f}", "halo_info")
                         .glob("halo_info_*.asdf")
            ) 
            if m 
    ]
    if not files: 
        raise FileNotFoundError(f"no files available for simulation {simname!r} at z={redshift}")
    
    # Load halo data from the files and add to the shared catalog file.
    logger.info(f"found {len(files)} files for simulation {simname!r} at z={redshift}")    

    mass_bins = np.asarray(bins, dtype = 'f8') # bins is an array of numbers
    assert mass_bins.ndim == 1 and mass_bins.shape[0] > 2, "insufficient bins"
    
    # Load any one files and get the unit mass in Msun. Using this mass, the mass bins are
    # converted from actual mass in Msun to unitless particle numbers. 
    with asdf.open( files[0] ) as af: 
        unit_mass  = af["header"]["ParticleMassMsun"] # particle mass in Msun
        box_size   = af["header"]["BoxSizeMpc"]       # box size in Mpc

    # Histogram calculation:
    if nprocs < 1: nprocs = os.cpu_count()
    logger.info(f"counting mass from {len(files)} files, distributed on {nprocs} processes...")
    with multiprocessing.Pool(processes = nprocs) as pool:
        count_hist = reduce( 
            np.add, 
            pool.starmap(
                _histogram, 
                zip( 
                    map(
                        lambda fn: CompaSOHaloCatalog(fn, cleaned = False, fields = ["N"]).halos["N"], 
                        files,
                    ), 
                    repeat(mass_bins / unit_mass), # bin edges in particle mass unit
                )
            ), 
            0, 
        )
    logger.info(f"mass counting completed")

    nonzero_mask, = np.where(count_hist > 0.) # mask for non-empty bins
    count_hist    = count_hist[nonzero_mask]

    # Bin centers (geometric):
    centers = np.sqrt( mass_bins[1:] * mass_bins[:-1] )[nonzero_mask] 
    
    # Logarithmic bin widths:
    widths = np.diff( np.log(mass_bins) )[nonzero_mask] 
    
    # Halo mass-function: counts per unit log mass bin per unit volume:
    dndlnm = count_hist / ( box_size**3 * widths ) # unit: Mpc^-3
    if smooth > 1:
        # Smoothing the mass-function using a boxcar window: 
        smooth = min(int(smooth), len(dndlnm) // 4 ) 
        window = np.ones( smooth ) / smooth
        dndlnm = np.convolve(dndlnm, window, mode = "same")

    # Saving the data as a plain text file (optional: only if path specified)
    if outfile:
        logger.info(f"saving data to file: {str(outfile)!r}")
        np.savetxt( 
            outfile, 
            np.log([ centers, dndlnm ]).T, 
            header = '\n'.join([
                f"{attr}: {value}" for attr, value in [
                    ( "SimName"         , simname   ),
                    ( "Redshift"        , redshift  ),
                    ( "BoxsizeMpc"      , box_size  ),
                    ( "ParticleMassMsun", unit_mass )
                ]
            ]), 
        )

    return (dndlnm, centers)

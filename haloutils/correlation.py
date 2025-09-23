
import numpy as np
import os, re, tempfile, shutil, multiprocessing, logging
from pathlib import Path
from collections import namedtuple
from typing import Any
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

pcargs_t = namedtuple("pcargs_t", ["boxsize", "D1", "D2", "R1", "R2", "rbins", "meta"])
pcres_t  = namedtuple("pcres_t" , ["D1D2", "D1R2", "D2R1", "R1R2", "size_D", "size_R"])

def abacus_corrfunc(
        simname  : str,
        redshift : float,
        rbins    : list[float], 
        mrange1  : tuple[float, float] = None,
        mrange2  : tuple[float, float] = None,
        outfile  : Path                = None, 
        workdir  : Path                = None, 
        loc      : Path                = None,
        nthreads : int                 = -1,
        subdivs  : int                 =  4,
        rseed    : int                 = None,
        randsize : float               =  3.,
    ) -> int:

    from abacusnbody.metadata import get_meta

    logger = logging.getLogger()
    
    success_code = -1 # indicate error: 0=success, 1=failure
    
    # Check the simulation details are valid. If yes, load boxsize and other 
    # metadata to include in the output file.
    try:
        meta = get_meta(simname, redshift)

        # Filter the metadata dict to contain only the selected fields
        h = meta["H0"] / 100. # hubble parameter in 100 km/s/Mpc
        meta = {
            "simname"          : meta["SimName"  ],
            "redshift"         : meta["Redshift" ],
            "H0"               : meta["H0"       ],
            "Om0"              : meta["Omega_M"  ],
            "Ode0"             : meta["Omega_DE" ],
            "Ob0"              : meta["omega_b"  ]/h**2, # converting Obh2 to Ob0
            "w0"               : meta["w0"       ],
            "wa"               : meta["wa"       ],
            "ns"               : meta["n_s"      ],
            "SODensity"        : meta["SODensity"], # first item is used as value of Delta_m
            "boxsize"          : meta["BoxSize"  ],
            "particleMassHMsun": meta["ParticleMassHMsun"],
            "boxsizeMpc"       : meta["BoxSizeMpc"       ],
            "particleMassMsun" : meta["ParticleMassMsun" ],
        }
    except ValueError:
        logger.error(f"data for simulation {simname!r} at z={redshift} not availbale.")
        return success_code
    
    if workdir is None:
        # No working directory is specified: create a temporary dir... 
        workdir  = Path( tempfile.mkdtemp(prefix="corrfn") )
        clean_up = True
    else:
        workdir  = Path( workdir )
        clean_up = False 

    try:
       
       # Setting up the arguments for pair counting:
       pcargs = _setup_abacus_workspace(
           simname, redshift, rbins, mrange1, mrange2, 
           workdir, loc, meta, nthreads, rseed, randsize,
       )

       # Count number of pairs between the datasets: this will count the
       # no. of pairs for full datasets, and jackknife subdivisions of the 
       # entire box (if enabled). 
       full_result, jack_results = count_pairs_jack(
           pcargs.D1, pcargs.D2, pcargs.R1, pcargs.R2, 
           rbins    = rbins, 
           subdivs  = subdivs,
           boxsize  = pcargs.boxsize, 
           periodic = True, 
           nthreads = nthreads
       )

       # TODO: Save data to specified path:

       success_code = 0

    finally: 
        if clean_up: shutil.rmtree(workdir) # clean-up workspace
        return success_code

def _setup_abacus_workspace(
        simname  : str,
        redshift : float,
        rbins    : list[float], 
        mrange1  : tuple[float, float],
        mrange2  : tuple[float, float],
        workdir  : Path, 
        loc      : Path,
        meta     : dict, 
        nthreads : int,
        rseed    : int,
        randsize : float,
    ) -> pcargs_t:

    from os.path import getsize
    from base64 import b64encode

    logger = logging.getLogger()
    
    # If catalogs to find the correlation exist in the current directory
    # (NOTE: in the correct format, that can be loaded as a numpy.memmap)
    # use those data. If not available, datafiles are created from the 
    # halo catalogs.

    def _hash(x) -> str:
        if x is None: return "full" # file conatining all mass values
        range_info_string = "{:.8e}:{:.8e}".format(*x)
        return b64encode(range_info_string.encode("utf-8")).decode("utf-8")

    fn1 = workdir.joinpath("abacus." + _hash(mrange1) + ".bin") # filename for dataset #1
    fn2 = workdir.joinpath("abacus." + _hash(mrange2) + ".bin") # filename for dataset #2 
    missing_fn1, missing_fn2 = not fn1.exists(), not fn2.exists()
    if missing_fn1 or missing_fn2: 

        # List all available halo catalogs:
        files = list(
            Path(loc).expanduser().absolute() 
                     .joinpath(simname, "halos", f"z{redshift:.3f}", "halo_info")
                     .glob("halo_info_*.asdf")
        )
        if not files: 
            raise FileNotFoundError(f"missing files in {str(loc)!r} for {simname!r}, z={redshift}")
        
        # Load missing data
        if missing_fn1:
            logger.info(f"getting data for mass range {mrange1}...")
            with open(fn1, 'wb') as fp, multiprocessing.Pool(processes = nthreads) as pool:
                for d1 in pool.map(_load_abacus, [ (file, mrange1) for file in files ]):
                    np.array(d1, copy = None).astype("f8").tofile(fp)
                del d1 # will be created later as a memory map to save space...
        
        if missing_fn2:
            logger.info(f"getting data for mass range {mrange2}...")
            with open(fn2, 'wb') as fp, multiprocessing.Pool(processes = nthreads) as pool:
                for d2 in pool.map(_load_abacus, [ (file, mrange2) for file in files ]):
                    np.array(d2, copy = None).astype("f8").tofile(fp)
                del d2 # will be created later as a memory map to save space...

    def memmap(fn: str):
        # Get the number of halos in the file from its file size. Assuming the
        # file is the catalog position array buffer saved using `tofile`...  
        FLOAT64_SIZE = 8 # No. of bytes for float64
        n_halos, __rem_bytes = divmod( getsize(fn), (3*FLOAT64_SIZE) )
        assert __rem_bytes == 0

        # Load as memmap
        arr = np.memmap(fn, dtype = "f8", mode = 'r', shape = (n_halos, 3)) 
        return arr, n_halos

    # Loading the data as memory maps
    d1, n_halo1 = memmap(fn1) 
    d2, n_halo2 = memmap(fn2)

    # Gnerating random catalog. This catalog is also saved into a binary file 
    # loaded as memory map... NOTE: Since both sets are from the same catalog, 
    # no second random catalog is needed. 
    rng     = np.random.default_rng(rseed)
    n_randx = int(max(n_halo1, n_halo2)*randsize) # size of the random catalog
    boxsize = meta["boxsize"] # in Mpc/h

    logger.info(f"generating random catalog of size {n_randx}...")
    rx  = rng.uniform(-boxsize / 2., boxsize / 2., size = (n_randx, 3))
    fnr = workdir.joinpath("abacus.rand.bin")
    rx.astype("f8").tofile(fnr)
    del rx

    # Loading random catalog as memory map
    rx, n_randx = memmap(fnr)

    logger.info("completed setting-up for abacus halo correlation.")
    return pcargs_t(boxsize, d1, d2, rx, None, rbins, meta)

def _load_abacus(args):

    file, mrange = args

    # Load catalog file
    catalog   = CompaSOHaloCatalog(file, cleaned = False, fields = ["SO_central_particle", "N"])
    unit_mass = catalog.header["ParticleMassMsun"] 
    
    mass = np.array(catalog.halos["N"])
    posn = np.array(catalog.halos["SO_central_particle"]) # halo poition coords in Mpc/h
    if mrange is not None:
        # Filtering mass range:
        na, nb = mrange[0] / unit_mass, mrange[1] / unit_mass     # to particle number
        posn   = np.array( posn[( na <= mass ) & ( mass < nb )] ) # copy data
        
    return posn

def count_pairs(
        D1       : Any,
        D2       : Any,
        R1       : Any,
        R2       : Any,
        rbins    : list[float], 
        boxsize  : float, 
        periodic : bool  = False,
        nthreads : int   = -1, 
    ) ->pcres_t:
    
    from Corrfunc.theory.DD import DD

    def _count_pairs(d1, d2=None, **kwargs):
        data = { f"{c}1": d1[:,i] for i, c in enumerate("XYZ") }
        if d2 is not None:
            data.update({ f"{c}2": d2[:,i] for i, c in enumerate("XYZ") })
            autocorr = 0 
        else:
            autocorr = 1
        count_result = DD(autocorr = autocorr, **data, **kwargs)
        return count_result["npairs"]

    logger   = logging.getLogger()
    settings = dict(
        nthreads = os.cpu_count() if nthreads < 1 else nthreads, 
        binfile  = rbins, 
        periodic = periodic,
        boxsize  = boxsize, 
        verbose  = False,
    )

    logger.info(f"counting pairs of D1 and R2...")
    d1r2 = _count_pairs(D1, R2, **settings)

    if D2 is D1 or D2 is None:
        logger.info(f"counting pairs of D1 and D2...")
        d1d2 = _count_pairs(D1, **settings)
        d2r1 = d1r2
    else:
        logger.info(f"counting pairs of D1 and D2...")
        d1d2 = _count_pairs(D1, D2, **settings)
        
        logger.info(f"counting pairs of D2 and R1...")
        d2r1 = _count_pairs(R1, D2, **settings)
    
    logger.info(f"counting pairs of R1 and R2...")
    if R2 is R1 or R2 is None:
        r1r2 = _count_pairs(R1, None, **settings)
    else:
        r1r2 = _count_pairs(R1, R2, **settings)

    retval = pcres_t(
        d1d2, 
        d1r2, 
        d2r1, 
        r1r2, 
        ( D1.shape[0], D2.shape[0] if D2 is not None else D1.shape[0] ),
        ( R1.shape[0], R2.shape[0] if R2 is not None else R1.shape[0] ),
    )
    return retval

def count_pairs_jack(
        D1       : Any,
        D2       : Any,
        R1       : Any,
        R2       : Any,
        rbins    : list[float], 
        subdivs  : int,
        boxsize  : float, 
        periodic : bool  = False,
        nthreads : int   = -1, 
        workdir  : Path  = None,
    ) -> tuple[pcres_t, list[pcres_t]]:

    logger = logging.getLogger()

    # Pairs counts for the full region:
    logger.info("counting pairs for the full box...")
    full_result = count_pairs(
        D1, D2, R1, R2, 
        rbins    = rbins, 
        boxsize  = boxsize, 
        periodic = periodic, 
        nthreads = nthreads,
    )
    if subdivs < 2: return full_result, []

    # Sub-dividing the box into sub-regions. Assign each point the catalog
    # with an integer index corresponding to the sub-region.     
    def subdivide(arr, boxsize: float, ndivs: int):
        i = np.floor(arr / (boxsize / ndivs)).astype("i8") # 3d sub-region index 
        i = np.clip(i, 0, ndivs-1) # set points outside to sub-regions at edges 
        i = i[:,0] + ndivs*( i[:,1] + ndivs*i[:,2] ) # flattening index
        return i
    
    n_subdivisions = subdivs**3 # no. of sub-regions
    
    # Assign sub-box indices
    D1_boxid = subdivide( D1, boxsize, subdivs )
    D2_boxid = subdivide( D2, boxsize, subdivs )
    R1_boxid = subdivide( R1, boxsize, subdivs )
    R2_boxid = subdivide( R2, boxsize, subdivs )

    def as_memmap(arr, dir, id):
        # This fill first save the arr array buffer as a binary file and then
        # return a memory map of the same array. This can be useful if the array
        # is very large nad there are many...  
        s   = (arr.shape[0], arr.shape[1])   # shape of the original array
        fn  = os.path.join(dir, f"{id}.bin") # filename for the memory map
        np.array(arr, copy = None).astype("f8").tofile(fn) # save the buffer
        arr = np.memmap(fn, dtype = np.float64, mode = 'r', shape = s)
        return arr, fn

    # Pair counts for the sub-regions:
    jack_results = []
    with tempfile.TemporaryDirectory(dir = workdir) as _tdir:
        for i in range(n_subdivisions):
            # Load the data for the i-th sub-region and count pairs:
            D1_box, fn0 = as_memmap( D1[D1_boxid != i], _tdir, 0 )
            D2_box, fn1 = as_memmap( D2[D2_boxid != i], _tdir, 1 )
            R1_box, fn2 = as_memmap( R1[R1_boxid != i], _tdir, 2 )
            R2_box, fn3 = as_memmap( R2[R2_boxid != i], _tdir, 3 )
            logger.info(f"counting pairs for sub-box {i}...")
            try:
                jack_result = count_pairs(
                    D1_box, D2_box, R1_box, R2_box, 
                    rbins    = rbins, 
                    boxsize  = boxsize, 
                    periodic = periodic, 
                    nthreads = nthreads,
                )
                jack_results.append(jack_result)
            finally:
                for fn in (fn0, fn1, fn2, fn3): os.remove(fn) # delete files
    
    return full_result, jack_results

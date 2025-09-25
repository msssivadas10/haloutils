
import numpy as np, numpy.typing as nt
import os, tempfile, shutil, multiprocessing, logging, asdf
from pathlib import Path
from collections import namedtuple
from typing import TypeVar, Literal, Callable
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

__version__ = "0.5a"

def correlation(
        D1D2      : nt.NDArray[np.int64], 
        R1R2      : nt.NDArray[np.int64], 
        D1R2      : nt.NDArray[np.int64], 
        D2R1      : nt.NDArray[np.int64], 
        estimator : Literal["natural", "davis-peebles", "hamilton", "landy-szalay"] = "natural", 
    ) -> nt.NDArray[np.float64]:
    # TODO: add docstring

    ids_for_natural         = ["natural", "peebles-hauser"]
    ids_for_davis_peebles   = ["davis-peebles", "dp"]
    ids_for_hamilton        = ["hamilton"]
    ids_for_landy_szalay    = ["landy-szalay", "ls"]
    estimators_requiring_DR = ids_for_davis_peebles + ids_for_hamilton + ids_for_landy_szalay
    estimators_requiring_RR = ids_for_natural + ids_for_hamilton + ids_for_landy_szalay

    DD, RR, DR = np.array(D1D2), None, None
    if D1R2 is not None or D2R1 is not None:
        if   D1R2 is None: DR = np.array(D2R1)
        elif D2R1 is None: DR = np.array(D1R2)
        else:
            DR = ( np.array(D1R2) + np.array(D2R1) ) / 2.
    elif estimator.lower() in estimators_requiring_DR:
        raise ValueError(f"estimator {estimator!r} requires DR pair counts")
    if R1R2 is not None:
        RR = np.array(R1R2)
    elif estimator.lower() in estimators_requiring_RR:
        raise ValueError(f"estimator {estimator!r} requires RR pair counts")
    
    if estimator.lower() in ids_for_natural      : return  DD / RR - 1
    if estimator.lower() in ids_for_davis_peebles: return  DD / DR - 1
    if estimator.lower() in ids_for_hamilton     : return (DD * RR) / (DR**2) - 1
    if estimator.lower() in ids_for_landy_szalay : return (DD - 2*DR + RR) / RR
    raise ValueError(f"Unknown estimator {estimator!r}")

# Basic Pair Counting Functions:

pcres_t = namedtuple("pcres_t" , ["D1D2", "D1R2", "D2R1", "R1R2", "ND1", "ND2", "NR1", "NR2"])
_Coords = TypeVar("_Coords", nt.NDArray[np.float64], np.memmap)
def count_pairs(
        D1       : _Coords,
        D2       : _Coords,
        R1       : _Coords,
        R2       : _Coords,
        rbins    : list[float], 
        boxsize  : float, 
        periodic : bool  = False,
        nthreads : int   = -1, 
    ) ->pcres_t:
    # TODO: add docstring
    
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
        D1D2 = d1d2, 
        D1R2 = d1r2, 
        D2R1 = d2r1, 
        R1R2 = r1r2, 
        ND1  = D1.shape[0], 
        ND2  = D2.shape[0] if D2 is not None else D1.shape[0],
        NR1  = R1.shape[0], 
        NR2  = R2.shape[0] if R2 is not None else R1.shape[0],
    )
    return retval

def count_pairs_jack(
        D1       : _Coords,
        D2       : _Coords,
        R1       : _Coords,
        R2       : _Coords,
        rbins    : list[float], 
        subdivs  : int,
        boxsize  : float, 
        periodic : bool  = False,
        nthreads : int   = -1, 
        workdir  : Path  = None,
    ) -> tuple[pcres_t, list[pcres_t]]:
    # TODO: add docstring

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
        i = np.floor((arr + boxsize / 2) / (boxsize / ndivs)).astype("i8") # 3d sub-region index 
        i = np.clip(i, 0, ndivs-1) # set points outside to sub-regions at edges 
        i = i[:,0] + ndivs*( i[:,1] + ndivs*i[:,2] ) # flattening index
        return i
    
    n_subdivisions = subdivs**3 # no. of sub-regions

    D1_D2_are_same = D2 is D1 or D2 is None
    R1_R2_are_same = R2 is R1 or R2 is None
    
    # Assign sub-box indices
    D1_boxid = subdivide( D1, boxsize, subdivs )
    R1_boxid = subdivide( R1, boxsize, subdivs )
    D2_boxid = subdivide( D2, boxsize, subdivs ) if not D1_D2_are_same else D1_boxid
    R2_boxid = subdivide( R2, boxsize, subdivs ) if not R1_R2_are_same else R1_boxid

    def as_memmap(arr, dir, mapped_files: set):
        # This fill first save the arr array buffer as a binary file and then
        # return a memory map of the same array. This can be useful if the array
        # is very large and there are many...  
        if arr.shape[0] < 64: return arr     # no memory map if array is small
        id  = len(mapped_files) 
        s   = arr.shape                      # shape of the original array
        fn  = os.path.join(dir, f"{id}.bin") # filename for the memory map
        np.array(arr, copy = None).astype("f8").tofile(fn) # save the buffer
        arr = np.memmap(fn, dtype = np.float64, mode = 'r', shape = s)
        mapped_files.add(fn)
        return arr

    # Pair counts for the sub-regions:
    jack_results = []
    with tempfile.TemporaryDirectory(dir = workdir) as _tdir:
        for i in range(n_subdivisions):
            # Load the data for the i-th sub-region and count pairs:
            files_to_delete = set()
            D1_box = as_memmap( D1[D1_boxid != i], _tdir, files_to_delete )
            R1_box = as_memmap( R1[R1_boxid != i], _tdir, files_to_delete )
            D2_box = as_memmap( D2[D2_boxid != i], _tdir, files_to_delete ) if not D1_D2_are_same else D1_box
            R2_box = as_memmap( R2[R2_boxid != i], _tdir, files_to_delete ) if not R1_R2_are_same else R1_box
            
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
                for fn in files_to_delete: os.remove(fn) # delete files

    return full_result, jack_results

# Helper Functions:

def _memmap(fn: str):
    # Get the number of points in the file from its file size. Assuming the
    # file is the catalog position array buffer saved using `tofile`...  
    from os.path import getsize

    FLOAT64_SIZE = 8 # No. of bytes for float64
    n_pts, __rem_bytes = divmod( getsize(fn), (3*FLOAT64_SIZE) )
    assert __rem_bytes == 0

    # Load as memmap
    arr = np.memmap(fn, dtype = "f8", mode = 'r', shape = (n_pts, 3)) 
    return arr, n_pts

def _hash(x, prefix: str = '', suffix: str = '') -> str:
    # Generate a hash value for the mass range.
    from math import log10, floor
    
    def _str(xi): 
        integer = floor( xi*10**(20 - floor(log10(xi))) )
        return str(integer)
    
    range_string = '-'.join([ _str(xi) for xi in x ]) if x else "all"
    return prefix + range_string + suffix

def _savebuffer(
        loader   : Callable[[str, tuple[float, float]], _Coords], 
        files    : list[str], 
        fn       : str, 
        mrange   : tuple[float, float], 
        nthreads : int, 
    ):
    # Save position data (returned by the `loader` function when called with items in 
    # files) as a memory mappable file.
    logger = logging.getLogger()
    
    def _format(x): return "full" if not x else "{:.3e} to {:.3e}".format(*x)
    
    logger.info(f"getting data for mass range {_format(mrange)}...")
    with open(fn, 'wb') as fp, multiprocessing.Pool(processes = nthreads) as pool:
        for data in pool.starmap(loader, [ (file, mrange) for file in files ]):
            np.array(data, copy = None).astype("f8").tofile(fp)
        del data # will be created later as a memory map to save space...
    return

def _generate_random(
        size    : int, 
        xrange  : tuple[float, float], 
        rng     : np.random.Generator, 
        workdir : Path, 
    ):
    # Generating random catalog. This catalog is saved to a binary file and loaded 
    # as a memory map. 
    logger = logging.getLogger()

    logger.info(f"generating random catalog of size {size}...")
    fn = workdir.joinpath("abacus.rand.bin")
    rng.uniform(*xrange, size = (size, 3)).astype("f8").tofile(fn)
    rx, _ = _memmap(fn)
    return rx

def _cleanup_workdir_after_use(fn):

    import inspect, functools

    params   = inspect.signature( fn ).parameters
    defaults = { p.name: p.default for p in params.values() if p.default is not p.empty }

    @functools.wraps(fn)
    def decorator(*args, **kwargs):
        argdict = defaults | dict( zip(params, args) ) | kwargs
        workdir, clean_up = argdict.pop("workdir", None), False
        if not workdir:
            workdir, clean_up  = tempfile.mkdtemp(prefix = "corrfn"), True
        try: 
            retval = fn(**argdict, workdir = workdir)
        finally:
            if clean_up: 
                shutil.rmtree(workdir)
        return retval
    
    return decorator

# Data Specific Functions:

@_cleanup_workdir_after_use
def _corrfunc(
        loader       : Callable[[str, tuple[float, float]], _Coords],
        fn_prefix    : str,
        file_pattern : str,
        mrange1      : tuple[float, float],
        mrange2      : tuple[float, float],
        boxsize      : float,
        loc          : Path,
        workdir      : Path, 
        rbins        : list[float], 
        nthreads     : int,
        subdivs      : int,
        rseed        : int,
        randsize     : float,
        outfile      : Path,
        meta         : dict, 
        pktab        : nt.NDArray[np.float64], 
        saved_rbins  : list[float] = None,
        missing_file_msg : str     = '' 
    ):

    workdir  = Path(workdir).expanduser().absolute()
    nthreads = int( nthreads if nthreads > 1 else os.cpu_count() )
    
    logger = logging.getLogger()

    # Setting up pair counting: if the catalogs are available in the current directory use 
    # them, otherwise load from the halo catalog files.
    fn1 = workdir.joinpath(_hash(mrange1, prefix = fn_prefix, suffix = ".bin")) # filename for dataset #1
    fn2 = workdir.joinpath(_hash(mrange2, prefix = fn_prefix, suffix = ".bin")) # filename for dataset #2 
    if not fn1.exists() or not fn2.exists(): 
        files = list( Path(loc).glob( file_pattern ) )
        if not files: 
            raise FileNotFoundError(missing_file_msg or f"missing catalog files in {str(loc)!r}")
        if not fn1.exists(): _savebuffer( loader, files, fn1, mrange1, nthreads )
        if not fn2.exists(): _savebuffer( loader, files, fn2, mrange2, nthreads ) # also handle fn1 = fn2 correctly

    # Loading the data as memory maps:
    d1, n_halo1 = _memmap(fn1) 
    d2, n_halo2 = _memmap(fn2)

    # Generating random catalog. NOTE: Since both sets are from the same catalog, 
    # no second random catalog is needed. 
    r1 = _generate_random(
        size    = int(max(n_halo1, n_halo2)*randsize), # size of the random catalog
        xrange  = ( -boxsize / 2., boxsize / 2. ), 
        rng     = np.random.default_rng(rseed),
        workdir = workdir
    )

    logger.info("completed setting-up for abacus halo correlation.")

    # Count number of pairs between the datasets: this will count the no. of pairs 
    # for full datasets, and jackknife subdivisions of the entire box (if enabled). 
    full_result, jack_results = count_pairs_jack(
        d1, d2, r1, None, 
        rbins    = rbins, 
        subdivs  = subdivs,
        boxsize  = boxsize, 
        periodic = True, 
        nthreads = nthreads
    )
    meta.update({
        "jackknifeSamples": len(jack_results),
        "mrange1"         : mrange1,
        "mrange2"         : mrange2,
    })
    
    # Saving data:
    with asdf.AsdfFile({
        "header": meta,
        "data"  : {
            "powerSpectrum"   : pktab,
            "rbins"           : rbins if saved_rbins is None else saved_rbins, 
            "pairCounts_full" : full_result._asdict(),
            "pairCounts_jack" : {
                _field: np.stack([ getattr(t, _field) for t in jack_results ], axis = 0) 
                    for _field in pcres_t._fields
            }, 
        }
    }) as af:
        logger.info(f"saving data to {str(outfile)!r}")
        af.write_to(
            str( Path(outfile).expanduser().absolute() ), 
            all_array_compression = 'zlib', 
        )
    return

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
    ):
    # TODO: add docstring

    from numpy.lib.recfunctions import structured_to_unstructured as s2u
    from abacusnbody.metadata import get_meta

    # Check the simulation details are valid. If yes, load boxsize and other 
    # metadata to include in the output file.
    meta = get_meta(simname, redshift)
    h = meta["H0"] / 100. # hubble parameter in 100 km/s/Mpc

    # Get the matter power spectrum data in the metadata: The table in the metadata 
    # is in the units h/Mpc and (Mpc/h)^3 and at a specific redshift. This is converted
    # to a log(k in Mpc^-1) vs log(P in Mpc^3) table and interpolated to given redshift.  
    pktab = np.log( s2u( np.array(meta["CLASS_power_spectrum"]) ) ) # at z_pk
    dplus = meta['GrowthTable'] # growth factor table
    z_target = meta["Redshift"]
    try: dz_target = dplus[z_target]
    except KeyError: # get the nearest value within machine precision
        dz_target = dplus[ next(z for z in dplus if np.allclose(z, z_target)) ]
    z_pk = meta['ZD_Pk_file_redshift'] # power spectrum calculated at this redshift
    try: dz_pk = dplus[z_pk]
    except KeyError: # get the nearest value within machine precision
        dz_pk = dplus[ next(z for z in dplus if np.allclose(z, z_pk)) ]
    pktab[:,0] += np.log(h) 
    pktab[:,1] -= np.log(h)*3  + 2*np.log(dz_target / dz_pk)

    # Filter the metadata dict to contain only the selected fields
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
    
    rbins = np.array(rbins, dtype = "f8") # value in Mpc
    dirs  = [  simname, "halos", f"z{redshift:.3f}", "halo_info"  ]
    _corrfunc(
        loader           = _load_abacus, 
        fn_prefix        = "abacus.", 
        file_pattern     = "halo_info_*.asdf", 
        mrange1          =  mrange1, 
        mrange2          =  mrange2, 
        boxsize          =  meta["boxsize"], # value in Mpc/h
        loc              =  Path(loc).expanduser().absolute().joinpath(*dirs), 
        workdir          =  workdir, 
        rbins            =  rbins*h, # value in Mpc/h
        nthreads         =  nthreads, 
        subdivs          =  subdivs, 
        rseed            =  rseed, 
        randsize         =  randsize, 
        outfile          =  outfile, 
        meta             =  meta, 
        pktab            =  pktab, 
        saved_rbins      =  rbins, 
        missing_file_msg =  f"missing files in {str(loc)!r} for {simname!r}, z={redshift}",
    )
    return

def _load_abacus(file: str, mrange: tuple[float, float]) -> _Coords:
    catalog   = CompaSOHaloCatalog(file, cleaned = False, fields = ["SO_central_particle", "N"])
    unit_mass = catalog.header["ParticleMassMsun"] 
    mass, pos = np.array(catalog.halos["N"]),  np.array(catalog.halos["SO_central_particle"]) # units: Msun/h, Mpc/h
    if mrange is not None:
        na, nb = mrange[0] / unit_mass, mrange[1] / unit_mass    # to particle number
        pos    = np.array( pos[( na <= mass ) & ( mass < nb )] ) # copy data
    return pos

def galaxy_corrfunc(
        loc      : Path,
        rbins    : list[float], 
        mrange1  : tuple[float, float] = None,
        mrange2  : tuple[float, float] = None,
        outfile  : Path                = None, 
        workdir  : Path                = None, 
        nthreads : int                 = -1,
        subdivs  : int                 =  4,
        rseed    : int                 = None,
        randsize : float               =  3.,
    ):
    # TODO: add docstring

    galaxy_path = Path(loc).expanduser().absolute().joinpath("galaxy_info")
    if not galaxy_path.exists():
        raise FileNotFoundError(f"path does not exist: {str(loc)}")
    
    files = list( galaxy_path.glob("galaxy_info_*.asdf") )
    if not files: 
        raise FileNotFoundError(f"missing catalog files in {str(loc)!r}")
    
    # Loading metadata
    with asdf.open(files[0]) as af: meta = af["header"]
    pkfile = galaxy_path.joinpath('powerspectrum.txt')
    pktab  = np.loadtxt(pkfile) if pkfile.exists() else []

    rbins = np.array(rbins, dtype = "f8") # value in Mpc
    _corrfunc(
        loader           = _load_galaxies, 
        fn_prefix        = "galaxy.", 
        file_pattern     = "galaxy_info_*.asdf", 
        mrange1          =  mrange1, 
        mrange2          =  mrange2, 
        boxsize          =  meta["boxsizeMpc"], # value in Mpc
        loc              =  galaxy_path, 
        workdir          =  workdir, 
        rbins            =  rbins,
        nthreads         =  nthreads, 
        subdivs          =  subdivs, 
        rseed            =  rseed, 
        randsize         =  randsize, 
        outfile          =  outfile, 
        meta             =  meta, 
        pktab            =  pktab, 
        saved_rbins      =  rbins, 
        missing_file_msg =  f"missing catalog files in {str(loc)!r}",
    )
    return

def _load_galaxies(file: str, mrange: tuple[float, float]) -> _Coords:
    with asdf.open(file) as af:
        galaxy_type = af["data"]["galaxyType"]
        posn = af["data"]["galaxyPosition"]
        if mrange:
            start,  = np.where( galaxy_type == b'c' )
            stop    = np.hstack([ start[1:], galaxy_type.shape[0] ])
            ma, mb  = mrange
            parent_halo_mass = af["data"]["galaxyMass"][start] # same as central galaxy mass, in Msun
            selection_mask   = (ma <= parent_halo_mass) & (parent_halo_mass < mb)
            start, stop      = start[selection_mask], stop[selection_mask]
            selection        = np.zeros_like(galaxy_type, dtype = bool)
            for i, j in zip(start, stop): 
                selection[i:j] = True
            posn = posn[selection]
    return np.array(posn)

def configure_default_logger(dir: str = None):

    import logging.config, re
    
    # Path to log files:
    dir = Path().cwd().joinpath("logs") if dir is None else Path(dir)     # path log folder
    fn  = dir.joinpath(re.sub(r"(?<=\.)py$", "log", Path(__file__).name)) # full path to log files
    os.makedirs(fn.parent, exist_ok = True)
    
    # Configure logging:
    logging.config.dictConfig({
        "version": 1, 
        "disable_existing_loggers": True, 
        "formatters": { 
            "default": { "format": "[ %(asctime)s %(levelname)s %(process)d ] %(message)s" }
        }, 
        "handlers": {
            "stream": {
                "level"    : "INFO", 
                "formatter": "default", 
                "class"    : "logging.StreamHandler", 
                "stream"   : "ext://sys.stdout"
            }, 
            "file": {
                "level"      : "INFO", 
                "formatter"  : "default", 
                "class"      : "logging.handlers.RotatingFileHandler", 
                "filename"   : fn, 
                "mode"       : "a", 
                "maxBytes"   : 10485760, # create a new file if size exceeds 10 MiB
                "backupCount": 4         # use maximum 4 files
            }
        }, 
        "loggers": { "root": { "level": "INFO", "handlers": [ "stream", "file" ] } }
    })
    return

if __name__ == "__main__":

    import warnings, inspect, click
    warnings.catch_warnings(action = "ignore") 

    @click.group
    @click.version_option(__version__, message = "%(prog)s v%(version)s")
    def cli(): 
        # TODO: add docstring
        configure_default_logger()
        
    def with_options(fn, /, **option_spec: tuple[list[str], str]):

        from typing import get_args, get_origin

        params  = inspect.signature(fn).parameters
        options = [] 
        for key in params:
            if key not in option_spec: continue
            opts, help = option_spec[key]
            p          = params[key]
            decls      = [ f"--{key}".replace('_', '-'), *opts ] 
            attrs      = {}
            optype     = p.annotation
            if get_origin(optype) is list:
                optype, = get_args(optype)
                attrs.update( multiple = True, envvar = p.name.upper() )
            elif get_origin(optype) is tuple: 
                optype = get_args(optype)
            elif optype is Path:
                if p.name in [ "workdir", "loc" ]: 
                    optype = click.Path(file_okay = False, exists = True) 
                else:
                    optype = click.Path()
            attrs.update( type = optype )
            if p.default is not p.empty: 
                attrs.update( default = p.default, required = False )
            else:
                attrs.update( required = True )
            options.append( click.option(*decls, **attrs, help = help) )
        for option_decorator in reversed(options): 
            fn = option_decorator(fn)
        return fn

    abacus_corrfunc = cli.command(
        with_options(
            abacus_corrfunc, 
            simname  = (["-s" ], "Name of simulation"              ),
            redshift = (["-z" ], "Redshift value"                  ),
            rbins    = (["-r" ], "Distance bin edges (Mpc)"        ),
            mrange1  = (["-m1"], "Mass range for first set (Msun)" ),
            mrange2  = (["-m2"], "Mass range for second set (Msun)"),
            outfile  = (["-o" ], "Path to output file"             ), 
            workdir  = (["-w" ], "Working directory"               ), 
            loc      = (["-l" ], "Path to catalog files"           ),
            nthreads = (["-n" ], "Number of threads to use"        ),
            subdivs  = (["-j" ], "Number of jackknife samples"     ),
            rseed    = (["-rs"], "Random seed"                     ),
            randsize = (["-f" ], "Random catalog size"             ),
        )
    )
    galaxy_corrfunc = cli.command(
        with_options(
            galaxy_corrfunc, 
            loc      = (["-l" ], "Path to catalog files"           ),
            rbins    = (["-r" ], "Distance bin edges (Mpc)"        ),
            mrange1  = (["-m1"], "Mass range for first set (Msun)" ),
            mrange2  = (["-m2"], "Mass range for second set (Msun)"),
            outfile  = (["-o" ], "Path to output file"             ), 
            workdir  = (["-w" ], "Working directory"               ), 
            nthreads = (["-n" ], "Number of threads to use"        ),
            subdivs  = (["-j" ], "Number of jackknife samples"     ),
            rseed    = (["-rs"], "Random seed"                     ),
            randsize = (["-f" ], "Random catalog size"             ),
        )
    )
    cli()

    # NOTE: for testing: will be removed later...
    # logging.basicConfig(level=logging.INFO)
    # os.makedirs('./xi_out', exist_ok=True)
    # abacus_corrfunc(
    #     'AbacusSummit_hugebase_c000_ph000', 3., np.logspace(-1, 2, 9), 
    #     (5e+12, 1e+13), (5e+12, 1e+13), '_data/x.asdf', 
    #     'xi_out', '_data/', -1, 2
    # )
    # with asdf.open("_data/x.asdf") as af:
    #     print(af['data'])

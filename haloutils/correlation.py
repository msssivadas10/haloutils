
import numpy as np, numpy.typing as nt
import os, tempfile, shutil, multiprocessing, logging, asdf
from pathlib import Path
from collections import namedtuple
from typing import TypeVar, Literal
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

__version__ = "0.1a"

def correlation(
        D1D2      : nt.NDArray[np.int64], 
        R1R2      : nt.NDArray[np.int64], 
        D1R2      : nt.NDArray[np.int64], 
        D2R1      : nt.NDArray[np.int64], 
        estimator : Literal["natural", "dp", "hamilton", "ls"] = "natural", 
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

pcargs_t = namedtuple("pcargs_t", ["boxsize", "D1", "D2", "R1", "R2", "rbins", "meta"])
pcres_t  = namedtuple("pcres_t" , ["D1D2", "D1R2", "D2R1", "R1R2", "ND1", "ND2", "NR1", "NR2"])

_3dPointsArray = TypeVar("_3dPointsArray", nt.NDArray[np.float64])
def count_pairs(
        D1       : _3dPointsArray,
        D2       : _3dPointsArray,
        R1       : _3dPointsArray,
        R2       : _3dPointsArray,
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
        D1       : _3dPointsArray,
        D2       : _3dPointsArray,
        R1       : _3dPointsArray,
        R2       : _3dPointsArray,
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

# Data Specific Functions:

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
    # TODO: add docstring

    from numpy.lib.recfunctions import structured_to_unstructured as s2u
    from abacusnbody.metadata import get_meta

    logger = logging.getLogger()
    
    success_code = -1 # indicate error: 0=success, 1=failure
    nthreads     = int( nthreads if nthreads > 1 else os.cpu_count() )
    
    # Check the simulation details are valid. If yes, load boxsize and other 
    # metadata to include in the output file.
    try:
        meta = get_meta(simname, redshift)
        h = meta["H0"] / 100. # hubble parameter in 100 km/s/Mpc

        # Get the matter power spectrum data in the metadata: as a table of 
        # log(k in Mpc^-1) vs log(P in Mpc^3). The table in the metadata is 
        # in the units h/Mpc and (Mpc/h)^3 and at a specific redshift. So, 
        # unit conversion and interpolation to current redshift applied on it.   
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
    except ValueError:
        logger.error(f"data for simulation {simname!r} at z={redshift} not availbale.")
        return success_code
    
    if workdir is None:
        # No working directory is specified: create a temporary dir... 
        workdir  = Path( tempfile.mkdtemp(prefix="corrfn") )
        clean_up = True
    else:
        workdir  = Path( workdir )
        clean_up = False # no clean-up if wrokdir is gieven: must do manually...

    rbins      = np.array(rbins, dtype = "f8") # in Mpc
    rbins_HMpc = rbins * h # in Mpc/h

    try:
        # Setting up the arguments for pair counting:
        pcargs = _setup_abacus_workspace(
            simname, redshift, rbins_HMpc, mrange1, mrange2, 
            workdir, loc, meta, nthreads, rseed, randsize,
        )

        # Count number of pairs between the datasets: this will count the
        # no. of pairs for full datasets, and jackknife subdivisions of the 
        # entire box (if enabled). 
        full_result, jack_results = count_pairs_jack(
            pcargs.D1, pcargs.D2, pcargs.R1, pcargs.R2, 
            rbins    = rbins_HMpc, 
            subdivs  = subdivs,
            boxsize  = pcargs.boxsize, 
            periodic = True, 
            nthreads = nthreads
        )
        meta["jackknifeSamples"] = len(jack_results)

        # Save data to specified path:
        with asdf.AsdfFile({
            "header": meta,
            "data"  : {
                "powerSpectrum"   : pktab,
                "rbins"           : rbins, 
                "pair_counts"     : full_result._asdict(),
                "pair_counts_jack": {
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

        success_code = 0

    finally: 
        if clean_up: 
            shutil.rmtree(workdir) # clean-up workspace
    
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

    fn1 = workdir.joinpath("abacus." + _hash(mrange1) + ".bin").expanduser().absolute() # filename for dataset #1
    fn2 = workdir.joinpath("abacus." + _hash(mrange2) + ".bin").expanduser().absolute() # filename for dataset #2 
    if not fn1.exists() or not fn2.exists(): 

        # List all available halo catalogs:
        files = list(
            Path(loc).expanduser().absolute() 
                     .joinpath(simname, "halos", f"z{redshift:.3f}", "halo_info")
                     .glob("halo_info_*.asdf")
        )
        if not files: 
            raise FileNotFoundError(f"missing files in {str(loc)!r} for {simname!r}, z={redshift}")
        
        def fmtrange(x): 
            return "full" if not x else "{:.3e} to {:.3e}".format(*x)
        
        # Load missing data
        if not fn1.exists():
            logger.info(f"getting data for mass range {fmtrange(mrange1)}...")
            with open(fn1, 'wb') as fp, multiprocessing.Pool(processes = nthreads) as pool:
                for d1 in pool.map(_load_abacus, [ (file, mrange1) for file in files ]):
                    np.array(d1, copy = None).astype("f8").tofile(fp)
                del d1 # will be created later as a memory map to save space...
        
        if not fn2.exists(): 
            # automatically drop this part if both ranges are same, since fn1 = fn2 then...
            logger.info(f"getting data for mass range {fmtrange(mrange2)}...")
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


# logging.basicConfig(level=logging.INFO)
# os.makedirs('./xi_out', exist_ok=True)
# abacus_corrfunc(
#     'AbacusSummit_hugebase_c000_ph000', 3., np.logspace(-1, 2, 9), 
#     (5e+12, 1e+13), (5e+12, 1e+13), '_data/x.asdf', 
#     'xi_out', '_data/', -1, 2
# )
# with asdf.open("_data/x.asdf") as af:
#     print(af['data'])

if __name__ == "__main__":

    import click, inspect, logging.config, re, warnings
    warnings.catch_warnings(action = "ignore") 

    @click.group
    @click.version_option(__version__, message = "%(prog)s v%(version)s")
    def cli(): 
        # TODO: add docstring
        
        # Path to log files:
        fn = Path().cwd().joinpath(
            "logs", 
            re.sub( r"(?<=\.)py$", "log", Path(__file__).name )
        ) # full path to log files
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

    def __decorate_options(fn, options: dict[str, tuple[list[str], str, dict]]):
        empty        = inspect._empty 
        options_list = []
        for arg, p in inspect.signature(fn).parameters.items():
            if arg not in options: continue
            opts, help_string, attrs = options[arg]
            if p.default is empty      : attrs["required"] = True
            elif "default" not in attrs: attrs["default" ] = p.default 
            if "type" not in attrs     : attrs["type"    ] = p.annotation 
            if "multiple" in attrs     : attrs["envvar"  ] = p.name.upper()
            options_list.append( 
                click.option(
                    f"--{arg}".replace('_', '-'), *opts, 
                    **attrs,
                    help = help_string, 
                ) 
            )
        for option in reversed(options_list): fn = option(fn)
        return fn

    Dir = click.Path(file_okay = False, exists = True)

    abacus_corrfunc = cli.command(
        __decorate_options(
            abacus_corrfunc, 
            {
                "simname" : ( ["-s" ], "Name of simulation"             , {}                     ),
                "redshift": ( ["-z" ], "Redshift value"                 , {}                     ),
                "rbins"   : ( ["-r" ], "Distance bin edges (Mpc)"       , {"type": float, "multiple": True}), 
                "mrange1" : ( ["-m1"], "Maa range for first set (Msun)" , {"type":(float, float)}),
                "mrange2" : ( ["-m2"], "Maa range for second set (Msun)", {"type":(float, float)}),
                "outfile" : ( ["-o" ], "Path to output file"            , {"type": click.Path() }), 
                "workdir" : ( ["-w" ], "Working directory"              , {"type": Dir          }), 
                "loc"     : ( ["-l" ], "Path to catalog files"          , {"type": Dir          }),
                "nthreads": ( ["-n" ], "Number of threads to use"       , {}                     ),
                "subdivs" : ( ["-j" ], "Number of jackknife samples"    , {}                     ),
                "rseed"   : ( ["-rs"], "Random seed"                    , {}                     ),
                "randsize": ( ["-f" ], "Random catalog size"            , {}                     ),
            }
        )
    )
    cli()
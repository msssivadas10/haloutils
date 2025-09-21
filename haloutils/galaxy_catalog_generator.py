# Galaxy catalog generator app: Generates galaxy catalogs based on halo catalogs. 
# Currently support only AbacusSummit simulation. 
#
# Usage  : python3 galaxy_catalog_generator.py [OPTIONS]
# Options: See main setion at the end or use --help option.
#   

__version__ = "0.1a"

import numpy as np
import numpy.ctypeslib as npct, ctypes as ct
import os, os.path, glob, re, logging, asdf, time, click
import threading, multiprocessing
from collections import namedtuple
from typing import Literal

# Container for halo model parameters:
hmargs_t = np.dtype([
    ( "lnm_min"   ,  "<f8" ),( "sigma_m",  "<f8" ),( "lnm0"      ,  "<f8" ),
    ( "lnm1"      ,  "<f8" ),( "alpha"  ,  "<f8" ),( "scale_shmf",  "<f8" ), 
    ( "slope_shmf",  "<f8" ),( "z"      ,  "<f8" ),( "H0"        ,  "<f8" ),
    ( "Om0"       ,  "<f8" ),( "Delta_m",  "<f8" ),( "dplus"     ,  "<f8" ),
], align = True)

# Halo buffer record struct:
halodata_t = np.dtype([("id", "<i8"),("pos", "<f8", 3),("mass", "<f8")])               

# Galaxy buffer record struct:
galaxydata_t = np.dtype([("id", "<i8"),("pos", "<f8", 3),("mass", "<f8"),("typ", "S1")]) 

# A tuple of arguments to the galaxy catalog generation library function:  
cgargs_t = namedtuple("cgargs_t", 
    ["halo_path", "glxy_path", "logs_path", "hmargs", "bbox", 
     "pktab", "pktab_size", "lnma", "lnmb", "sigmatab_size" , 
     "filt", "mrsc_table", "seed", "nthreads"               ]
)

# Types of arguments to the galaxy catalog generation library function:
cgargtypes = cgargs_t(
    halo_path     = ct.c_char_p, 
    glxy_path     = ct.c_char_p, 
    logs_path     = ct.c_char_p, 
    hmargs        = npct.ndpointer(hmargs_t, 0, flags="C_CONTIGUOUS"),
    bbox          = npct.ndpointer("f8"    , 2, flags="C_CONTIGUOUS"), 
    pktab         = npct.ndpointer("f8"    , 2, flags="C_CONTIGUOUS"), 
    pktab_size    = ct.c_int64, 
    lnma          = ct.c_double, 
    lnmb          = ct.c_double, 
    sigmatab_size = ct.c_int64, 
    filt          = ct.c_int, 
    mrsc_table    = npct.ndpointer("f8", 2, flags="C_CONTIGUOUS"),
    seed          = ct.c_int64,
    nthreads      = ct.c_int,
)

############################################################################################################
#                                             PREPARATION
############################################################################################################

def pack_arguments(
        work_dir      : str,                 # path to the working directory
        hmargs        : tuple,               # halo model parameters      
        bounding_box  : np.ndarray,          # simulation bounding box   
        mass_range    : tuple[float, float], # mass range for variance table
        var_tabsize   : int,                 # size of the variance table    
        filter_code   : int,                 # filter function integer code
        pktable       : np.ndarray,          # power spectrum table: lnk vs lnp
        rseed         : int,                 # seed value for random number generation
        nthreads      : int,                 # no. of threads to use
        data_generator: map,                 # halo data
    ) -> cgargs_t:
    # Pack the arguments for the galaxy generator function in the correct order and
    # type. Also, create the halo catalog buffer in the correct format. 

    bounding_box = np.array(bounding_box, "f8"    ); assert bounding_box.shape == (2, 3)
    pktable      = np.array(pktable     , "f8"    ); assert pktable.shape[1]   == 2
    hmargs       = np.array(hmargs      , hmargs_t)
    lnma, lnmb   = float(mass_range[0]), float(mass_range[1])
    var_tabsize  = int(var_tabsize)
    filter_code  = int(filter_code)
    nthreads     = os.cpu_count()   if nthreads < 1 else int(nthreads)
    rseed        = int( time.time() if rseed is None else rseed )

    def encode(string: str): 
        assert len(string) <= 1024, f"string length exceeds maximum limit: {string}"
        return string.encode("utf-8")

    # Packing the arguments in correct order:
    args = cgargs_t(
        halo_path     = encode(os.path.join(work_dir, f"hbuf.bin")), # Path to the halo catalog file (input) 
        glxy_path     = encode(os.path.join(work_dir, f"gbuf.bin")), # Path to the galaxy catalog file (output)
        logs_path     = encode(os.path.join(work_dir, f"log"     )), # Path to the log file
        hmargs        = hmargs, 
        bbox          = bounding_box, 
        pktab         = pktable, 
        pktab_size    = pktable.shape[0], 
        lnma          = lnma, 
        lnmb          = lnmb, 
        sigmatab_size = var_tabsize, 
        filt          = filter_code, 
        mrsc_table    = np.zeros((var_tabsize, 4), "f8"),
        seed          = rseed, 
        nthreads      = nthreads
    ) 
    
    # Save the halo catalog data a binary file for sharing. Passing the halo catalog 
    # data loder as a generator, so that only the needed the data is loaded, making
    # effiicient use of memory. 
    with open(args.halo_path, 'wb') as fp: 
        # Data written as a stream of `halodata_t` structs.
        for halo_id, halo_pos, halo_mass in data_generator:
            n_halos     = np.size(halo_id)
            halo_buffer = np.empty((n_halos, ), dtype = halodata_t)
            halo_buffer["id"  ] = np.array(halo_id  ).astype("<i8", copy = False)
            halo_buffer["pos" ] = np.array(halo_pos ).astype("<f8", copy = False)
            halo_buffer["mass"] = np.array(halo_mass).astype("<f8", copy = False)
            halo_buffer.tofile(fp)
    
    return args

def prepare_abacus_workspace(
        work_dir : str,   # path to the working directory
        args     : dict,  # Values for halo model and other parameters
        simname  : str,   # Name of the simulation
        redshift : float, # Redshift 
        loc      : str,   # Path to look for halo catalog files
        rseed    : int,   # seed value for random number generation
        nthreads : int,   # no. of threads to use
    ) -> tuple[dict, cgargs_t]:
    # Prepare the arguments and metadata for catalog generation using abacus summit 
    # halo catalogs.     
    
    from math import exp
    from numpy.lib.recfunctions import structured_to_unstructured
    from abacusnbody.metadata import get_meta
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    # Look-up table to get `sigma_8` values used in the abacus summit cosmologies. 
    # This data is not in the catalog headers, so it is taken from documentations
    # <https://abacussummit.readthedocs.io/en/latest/cosmologies.html>.
    sigma8 = {
        "000": 0.807952, "001": 0.776779, "002": 0.808189, "003": 0.855190, "004": 0.749999, "009": 0.811362, 
        "010": 0.823630, "011": 0.905993, "012": 0.827899, "013": 0.813715, "014": 0.800000, "015": 0.835005, 
        "016": 0.793693, "017": 0.815903, "018": 0.819708, "019": 0.805050, "020": 0.811350, "021": 0.849842,   
        "022": 0.824961, "100": 0.808181, "101": 0.808156, "102": 0.808270, "103": 0.808075, "104": 0.808166, 
        "105": 0.808177, "106": 0.808181, "107": 0.808168, "108": 0.808177, "109": 0.808161, "110": 0.808170, 
        "111": 0.808179, "112": 0.824120, "113": 0.792107, "114": 0.808245, "115": 0.808089, "116": 0.866133, 
        "117": 0.808205, "118": 0.808152, "119": 0.808168, "120": 0.808181, "121": 0.808169, "122": 0.808170,   
        "123": 0.808176, "124": 0.808175, "125": 0.811995, "126": 0.803908, "130": 0.711201, "131": 0.807866, 
        "132": 0.808189, "133": 0.905163, "134": 0.808458, "135": 0.808160, "136": 0.882475, "137": 0.729693, 
        "138": 0.821432, "139": 0.793003, "140": 0.772033, "141": 0.799575, "142": 0.707082, "143": 0.847522, 
        "144": 0.891360, "145": 0.801404, "146": 0.762696, "147": 0.777157, "148": 0.714913, "149": 0.824854, 
        "150": 0.937655, "151": 0.860820, "152": 0.677885, "153": 0.794389, "154": 0.838698, "155": 0.735302, 
        "156": 0.801974, "157": 0.872315, "158": 0.829816, "159": 0.718521, "160": 0.876756, "161": 0.793066, 
        "162": 0.779589, "163": 0.838824, "164": 0.774159, "165": 0.835954, "166": 0.837463, "167": 0.768419, 
        "168": 0.871407, "169": 0.777925, "170": 0.716059, "171": 0.852878, "172": 0.765650, "173": 0.763962, 
        "174": 0.840113, "175": 0.708760, "176": 0.892483, "177": 0.806026, "178": 0.791239, "179": 0.775969, 
        "180": 0.894071, "181": 0.730036, 
    }

    logger = logging.getLogger()

    # Check if the simulation is available. If available, load the corresponding 
    # metadata.
    try:
        meta = get_meta(simname, redshift)
    except ValueError:
        logger.error(f"data for simulation {simname!r} at z={redshift} not availbale.")
        return
    except Exception:
        logger.exception(f"error getting metadata for simulation {simname!r} at z={redshift}")
    
    logger.info(f"using power spectrum and growth factor available with the metadata...") 
    
    # Linear power spectrum in the metadata is generated by CLASS at a specific 
    # redshift (`ZD_Pk_file_redshift` in the metadata). This data is stored as an 
    # astropy Table as k (h/Mpc) and P (Mpc/h)^3. 
    pktab = structured_to_unstructured(np.array(meta["CLASS_power_spectrum"])) # at z_pk

    # Growth factor table: since this is a dict of float->float, exact key matching
    # may not be possible because of precision issues. 
    dplus_tab = meta['GrowthTable'] 
    
    # Current redshift and corresponding growth
    z_target = meta["Redshift"]
    try:
        dz_target = dplus_tab[z_target]
    except KeyError: # get the nearest value within machine precision
        dz_target = dplus_tab[ next(z for z in dplus_tab if np.allclose(z, z_target)) ]

    # Redshift at which power spectrum is calculated and corresponding growth
    z_pk = meta['ZD_Pk_file_redshift']
    try:
        dz_pk = dplus_tab[z_pk]
    except KeyError: # get the nearest value within machine precision
        dz_pk = dplus_tab[ next(z for z in dplus_tab if np.allclose(z, z_pk)) ]
    
    # Interpolating the power spectrum table to the current redshift using the growth
    # factors. Also, convetring the table to log format: 
    h          = meta["H0"] / 100.
    pktab[:,0] = np.log(pktab[:,0]) + ( np.log(h) )   
    pktab[:,1] = np.log(pktab[:,1]) - ( np.log(h)*3  + 2*np.log(dz_target / dz_pk) )

    # Growth factor at current redshift, w.r.to present (z=0)
    dplus_at_z = dz_target / dplus_tab[0.]

    # Halo model arguments as tuple: NOTE: field order must be same as the hmargs_t struct
    hmargs = (
        args["lnm_min"   ]   , # lnm_min     
        args["sigma_m"   ]   , # sigma_m     
        args["lnm0"      ]   , # lnm0        
        args["lnm1"      ]   , # lnm1       
        args["alpha"     ]   , # alpha       
        args["scale_shmf"]   , # scale_shmf  
        args["slope_shmf"]   , # slope_shmf  
        meta["Redshift"  ]   , # z          
        meta["H0"        ]   , # H0          
        meta["Omega_M"   ]   , # Om0         
        meta["SODensity" ][0], # Delta_m     
        dplus_at_z           , # dplus      
    )

    # Simulation bounding box (values in Mpc): [-boxsize/2, boxsize/2]
    bounding_box_Mpc = [[ -0.5*meta["BoxSizeMpc"] ]*3, [ 0.5*meta["BoxSizeMpc"] ]*3]

    # Setting up the mass range for calculating the matter variance. Using the range
    # (unit_mass, 10^6*unit_mass) as a practical choice, with a default size of 101.
    lnma, lnmb = np.log(meta["ParticleMassMsun"]) + np.log([1., 1e+06])
    table_size = args.get("sigma_size", 101)
    filter_id  = { "tophat": 0, "gauss": 1 }.get( args.get("filter", "tophat") )

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
        "Growth"           : dplus_at_z,
        "sigma8"           : sigma8[re.search(r"c(\d{3})", simname).group(1)], # sigma8 parameter
        "Tcmb0"            : 2.728, # microwave background temperature in K
        "HaloModel"        : {      # halo model parameters
            "Mmin"     : exp(args["lnm_min"   ]),
            "sigmaM"   :     args["sigma_m"   ] ,
            "M0"       : exp(args["lnm0"      ]),
            "M1"       : exp(args["lnm1"      ]),
            "alpha"    :     args["alpha"     ] ,
            "scaleSHMF":     args["scale_shmf"] ,
            "slopeSHMF":     args["slope_shmf"] ,
        },
    } 

    # Filter out only the files with names matchng the pattern and sorting based on 
    # the integer index: 
    files = { 
        int( m.group(1) ): m.string 
            for m in map( 
                lambda fn: re.search(r"halo_info_(\d{3}).asdf", fn), # for files like `halo_info_123.asdf`
                glob.glob( 
                    os.path.join(
                        os.path.abspath(os.path.expanduser(loc)), # full path to files location 
                        simname, "halos", f"z{redshift:.3f}", "halo_info", 
                        "halo_info_*.asdf", 
                    ) 
                )
            ) 
            if m 
    }
    files  = [ files[idx] for idx in sorted(files) ]
    if not files: 
        logger.info(f"no files available for simulation {simname!r} at z={redshift}")
        return
    
    # Load halo data from the files and add to the shared catalog file.
    logger.info(f"found {len(files)} files for simulation {simname!r} at z={redshift}")

    def load_abacus_halo_catalog(fn: str):
        logger.info(f"loading halo catalog from file: {fn!r}")
        catalog = CompaSOHaloCatalog(fn, cleaned = False, fields = ["id", "SO_central_particle", "N"])
        h, unit_mass_Msun = catalog.header["H0"] / 100., catalog.header["ParticleMassMsun"] 
        uid  = catalog.halos["id"]                      # halo ID  
        pos  = catalog.halos["SO_central_particle"] / h # halo position coordinates in Mpc
        mass = catalog.halos["N"] * unit_mass_Msun      # halo mass in Msun
        return uid, pos, mass
    
    # Save the values as shared data: 
    cgargs = pack_arguments(
        work_dir, 
        hmargs, 
        bounding_box_Mpc, 
        (lnma, lnmb), 
        table_size, 
        filter_id, 
        pktab, 
        rseed, 
        nthreads,
        map(load_abacus_halo_catalog, files), 
    )
    return meta, cgargs

def prepare_workspace(
        work_dir : str,   # path to the working directory 
        args     : dict,  # Values for halo model and other parameters
        simname  : str,   # Name of the simulation
        redshift : float, # Redshift 
        loc      : str,   # Path to look for halo catalog files
        rseed    : int,   # seed value for random number generation
        nthreads : int,   # no. of threads to use
    ) -> tuple[dict, cgargs_t]:
    # Setting up shared data based on simulation. Type of simulation is calculated
    # based on the prefix of its name.
    if simname.startswith("AbacusSummit"):
        return prepare_abacus_workspace(work_dir, args, simname, redshift, loc, rseed, nthreads)
    raise RuntimeError(f"simulation {simname!r} is not supported")

############################################################################################################
#                                           DATA GENERATION
############################################################################################################

def generate_galaxies(args: cgargs_t) -> int:
    # Generate galaxies using shared halo catalog and parameters. Call this only  
    # after creating the shared files. This will return 0 on success and non-zero
    # on failure. 
    
    # Loading the shared library and setting up the functions
    lib = npct.load_library("libhaloutils", os.path.dirname(__file__))
    lib.cgenerate_galaxy_catalog.argtypes = [*cgargtypes, ct.POINTER(ct.c_int)]
    lib.cgenerate_galaxy_catalog.restype  = None

    assert os.path.exists( args.halo_path ), "missing shared halo catalog data"    

    # Start a thread for logging
    stop_event = threading.Event()
    logs_file  = args.logs_path
    thread     = threading.Thread(target = _log_watcher, args = (logs_file, stop_event), daemon = True)
    thread.start()
    
    # Galaxy generation
    error_flag = ct.c_int(1) # Error code: non-zero on error 
    lib.cgenerate_galaxy_catalog(*args, ct.byref(error_flag))

    if error_flag.value != 0:
        logger = logging.getLogger()
        logger.error(f"galaxy catalog generation failed with code {error_flag.value}")
        stop_event.set() # setting the stop event to end the thread
        thread.join()
        return -1

    # Wait for logging thread to finish
    thread.join()

    return 0

def _log_watcher(logs_file: str, stop_event: threading.Event) -> None:
    # Watch the log file, pass the lines to main log file as they appear.

    ONE_DAY      = 86400 # Seconds in a day
    LOG_SENTINAL = "END" # Sentinal value indicating the end of log buffer
    
    # Wait until the file is created, with timeout of one day 
    start_time = time.time()
    while not os.path.exists(logs_file):
        if stop_event.is_set(): return
        if time.time() - start_time > ONE_DAY: # set a wait timeout of 1 day
            import warnings
            warnings.warn("reached timeout for waiting log file: no logs will be written")
            return 
        time.sleep(0.1)

    # Redirecting messages from log buffer to main logger
    with open(logs_file, "r") as f:
        logger     = logging.getLogger()
        start_time = time.time()
        while not stop_event.is_set():
            message = f.readline().strip()
            if not message: 
                if time.time() - start_time > ONE_DAY: # set a wait timeout of 1 day
                    import warnings
                    warnings.warn("reached timeout for waiting log message: exiting")
                    return 
                time.sleep(0.1) # wait until next line is get written
                continue
            if message == LOG_SENTINAL: break # end of the file

            # Write log record with correct level:
            level, _, message = message.partition(':')
            if level.strip() == "error": logger.error(message) 
            else : logger.info(message) 
            
            # Reset the wait time, if message is recieved
            start_time = time.time() 

    return 

############################################################################################################
#                                         DATA EXPORT TO ASDF
############################################################################################################

def export_data_products(
        cgargs : cgargs_t, # Contains output filename and extra data
        meta   : dict,     # Metadata to include in the ASDF files as header
        path   : str ,     # Path to the output folder - files will be in galaxy_info subdir.
    ) -> None:
    # Export data from the output buffer to ASDF file(s) in the given path. Also save
    # some additional data products such as matter power spectrum and variance tables, 
    # which are useful for later processing.  

    from math import ceil

    GBUF_REC_SIZE = galaxydata_t.itemsize # Size of a galaxy buffer record
    FILE_MIN_SIZE = 1073741824            # Minimum size limit for an output data chunk (=1 GiB)

    logger = logging.getLogger()

    galaxy_file = cgargs.glxy_path
    assert os.path.exists(galaxy_file) and os.path.getsize(galaxy_file) % GBUF_REC_SIZE == 0  # file check

    # Folder for saving output ASDF files (create if not exist)
    outdir = os.path.join( os.path.abspath(os.path.expanduser(path)), "galaxy_info" )
    os.makedirs(outdir, exist_ok = True)
    logger.info(f"saving galaxy catalog files to {outdir!r}...")
    
    # Loading galaxy data from the binary file and save them in ASDF files: 
    # Each file will store the metadata (header), and the data section contains 
    # position (Mpc), mass (Msun), parent halo ID and galaxy type (c for central, 
    # s for satellite).
    total_items = os.path.getsize(galaxy_file) // GBUF_REC_SIZE # total numebr of records
    chunk_size  = ceil(FILE_MIN_SIZE / GBUF_REC_SIZE)           # size of chunk of data (min: 1 GiB)

    args, files_count, n_items = [], 0, 0
    while n_items < total_items:
        start = files_count * chunk_size             # start of the block
        count = min(chunk_size, total_items - start) # size of the block
        fn    = os.path.join(outdir, f"galaxy_info_{files_count:03d}.asdf") # output file 
        args.append((galaxy_file, start, count, fn, meta))
        n_items     += count
        files_count += 1

    # Distributed file export:
    logger.info(f"exporting data to asdf format ({cgargs.nthreads} processes)...")
    with multiprocessing.Pool(processes = cgargs.nthreads) as pool:
        file_summary = pool.map(_write_asdf_chunck, args)
    logger.info(f"written {files_count} galaxy catalog files.")

    # Writing a summary file:
    with open(os.path.join(outdir, "summary.txt"), 'w') as fp: 
        fp.write( "file_name, central_count, satellite_count, file_size_bytes \n" )
        for line in file_summary: fp.write(line + '\n')
        logger.info(f"written summary file: {fp.name!r}")

    # Writing extra data products: matter power spectrum and halo mass - variance - halo 
    # concentration table for later uses. These data are send back by the catalog generation
    # program by re-writing the main pipe
    with open(os.path.join(outdir, "powerspectrum.txt"), 'w') as fp:
        np.savetxt(fp, cgargs.pktab, header = "log_wavenum, log_power")
        logger.info(f"written matter power spectrum data to file: {fp.name!r}")

    with open(os.path.join(outdir, "halodata.txt"), 'w') as fp:
        np.savetxt(fp, cgargs.mrsc_table, header = "log_mass, log_radius, log_sigma, log_conc")
        logger.info(f"written halo data to file: {fp.name!r}")

    return

def _write_asdf_chunck(args: tuple[str, int, int, str, dict]) -> str:
    # Write a part of the data as ASDF file. 
    
    galaxy_file, start, count, fn, meta = args

    # Loading the galaxy data buffer as a memmap
    mm = np.memmap(galaxy_file, dtype = galaxydata_t, mode = 'r', offset = 0)
    galaxy_buffer = np.array( mm[start:start+count] ) 

    # Calculating galaxy counts:
    types, counts = np.unique(galaxy_buffer["typ"], return_counts = True)
    counts_dict   = dict(zip(types, counts))
    assert set(counts_dict) == { b'c', b's' }

    with asdf.AsdfFile({
        "header" : meta, 
        "data"   : {
            "parentHaloID"   : galaxy_buffer["id"  ], 
            "galaxyPosition" : galaxy_buffer["pos" ], 
            "galaxyMass"     : galaxy_buffer["mass"], 
            "galaxyType"     : galaxy_buffer["typ" ],
        } 
    }) as af:
        af.write_to(fn, all_array_compression = 'zlib')

    # Prepare a summary of the current file: 
    summary = ', '.join([ 
        os.path.basename(fn)     , # basename of the current file 
        str( counts_dict[b'c']  ), # central galaxies in this file 
        str( counts_dict[b's']  ), # satellite galaxies in this file
        str( os.path.getsize(fn)), # file size in bytes
    ])   
    return summary

############################################################################################################
#                                                MAIN
############################################################################################################

@click.version_option(__version__, message = "%(prog)s v%(version)s")
@click.option("--simname"     , help="Name of simulation"             , required=True    , type=str                              )
@click.option("--redshift"    , help="Redshift value"                 , required=True    , type=float                            )
@click.option("--mmin"        , help="Central galaxy threshold mass"  , required=True    , type=float                            )
@click.option("--m0"          , help="Satellite galaxy threshold"     , required=True    , type=float                            )
@click.option("--m1"          , help="Satellite count amplitude"      , required=True    , type=float                            )
@click.option("--sigma-m"     , help="Central galaxy width parameter" ,  default=0.      , type=float                            )
@click.option("--alpha"       , help="Satellite power-law count index",  default=1.      , type=float                            )
@click.option("--scale-shmf"  , help="SHMF scale parameter"           ,  default=0.5     , type=float                            )
@click.option("--slope-shmf"  , help="SHMF slope parameter"           ,  default=2.      , type=float                            )
@click.option("--filter-fn"   , help="Filter function for variance"   ,  default="tophat", type=click.Choice(["tophat", "gauss"]))
@click.option("--sigma-size"  , help="Size of variance table"         ,  default=101     , type=int                              )
@click.option("--output-path" , help="Path to output files"           ,  default='.'     , type=click.Path(file_okay = False)    )
@click.option("--catalog-path", help="Path to catalog files"          ,  default='.'     , type=click.Path(exists    = True )    )
@click.option("--nthreads"    , help="Number of threads to use"       ,  default=-1      , type=int                              )
def galaxy_catalog_generator(
        simname      : str,   
        redshift     : float,
        mmin         : float, 
        m0           : float,
        m1           : float,
        sigma_m      : float =  0. ,                          
        alpha        : float =  1. ,                          
        scale_shmf   : float =  0.5,
        slope_shmf   : float =  2. ,
        filter_fn    : Literal["tophat", "gauss"] = "tophat",
        sigma_size   : int   = 101 ,
        output_path  : str   = '.' ,
        catalog_path : str   = '.' ,
        nthreads     : int   = -1  ,
        rseed        : int   = None,
    ) -> None:
    """
    Generate galaxy catalogs based on a halo catalog and halo model.\f 

    Parameters
    ----------

    simname : str
        Name of the simulation. Currently, only abacus summit simulations have support.

    redshift : float
        Redshift value 

    mmin : float
        Threshold mass for central galaxy formation, `M_min` (unit: Msun). 

    m0 : float
        Threshold mass for satellite galaxy formation, `M0` (unit: Msun).
    
    m1 : float
        Amplitude parameter for satellite count, `M1` (unit: Msun).   
    
    sigma_m : float, default=0
        Width parameter for central count, `sigma_M`. If the value is 0, then a step 
        function is used instead of the general sigmoid / smooth-step function.

    alpha : float, default=1
        Index parameter for satellite count, `alpha` 

    scale_shmf : float, default=0.5
        Scale parameter for the subhalo mass-function. 

    slope_shmf : float, default=2
        Slope parameter for the subhalo mass-function.

    filter_fn : {tophat, gauss}, optional
        Filter function used for variance calculations.

    sigma_size : int, default=101
        Size of the variance table calculated from the power spectrum table.

    output_path : Path, optional
        Path to save output files. Output ASDF files and metadata file can be 
        found in the `galaxy_info` subdirectory. Use current directory as default. 
        Files saved will be 

        - `galaxy_info_{xyz}.asdf` - galaxy catalogs. `xyz` is a 3-digit index of
          the catalog starting from `000`. 

        -  `metadata.asdf` - other parameters and values.

    catalog_path : Path, optional
        Path to search halo catalogs (directory structure is simulation specific). 
        Use current directory as default.

    nthreads : int, optional
        Number of threads, default is to use the number returned by `os.cpu_count`. 

    rseed : int, optional
        Random seed value. Must be an integer. (It can also be specifed by the 
        environment variable `RSEED`) 

    Notes
    -----
    To use this as a CLI tool, for supported simulations, use

    ```
    python -m haloutils.galaxy_catalog_generator OPTION1=VALUE1 ...
    ```

    where the options are same as the parameters (with prefix `--` and `_` in names 
    replaced by `-`).  

    """

    import shutil, tempfile
    from math import log
    
    # Random seed value can also be passed as an environment variable. It should
    # be a positive integer. Otherwise, a default value is generated based on the
    # current time. 
    if not isinstance(rseed, int):
        rseed = os.environ.get("RSEED", "None")
        rseed = int(rseed) if str.isnumeric(rseed) else None

    # Creating a temporary working directory: all the intermediate files like 
    # halo and galaxy data buffers are created in this temp directory.
    work_dir = tempfile.mkdtemp(prefix = f".gcg.{os.getpid()}.")

    # Setting up shared data based on simulation. Type of simulation is calculated
    # based on the prefix of its name.  
    meta_and_args = prepare_workspace(
        work_dir, 
        {
            "lnm_min"   : log(mmin) , "sigma_m"   : sigma_m   ,
            "lnm0"      : log(m0)   , "lnm1"      : log(m1)   ,
            "alpha"     : alpha     , "scale_shmf": scale_shmf,
            "slope_shmf": slope_shmf, "sigma_size": sigma_size,
            "filter"    : filter_fn ,
        }, 
        simname, 
        redshift, 
        catalog_path, 
        rseed, 
        nthreads,
    )
    if not meta_and_args: return # catalog generation failed :(
    meta, args = meta_and_args
    
    # Generating the galaxy data and exporting to ASDF format
    generate_galaxies(args)  
    export_data_products(args, meta, output_path) 

    # Cleaning up the working directory:
    shutil.rmtree(work_dir, ignore_errors = True)
    return

if __name__ == "__main__":

    import logging.config, warnings
    warnings.catch_warnings(action = "ignore")
    
    def get_log_filename(): 
        p  = os.path.join( os.getcwd(), "logs" )
        os.makedirs(p, exist_ok = True)
        return os.path.join( p, re.sub(r"(?<=\.)py$", "log", os.path.basename(__file__)) )
    
    # Configure logging
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
                "filename"   : get_log_filename(), 
                "mode"       : "a", 
                "maxBytes"   : 10485760, # create a new file if size exceeds 10 MiB
                "backupCount": 4         # use maximum 4 files
            }
        }, 
        "loggers": { "root": { "level": "INFO", "handlers": [ "stream", "file" ] } }
    })

    galaxy_catalog_generator = click.command( galaxy_catalog_generator )
    galaxy_catalog_generator()

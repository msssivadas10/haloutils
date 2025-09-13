# Galaxy catalog generator app: Generates galaxy catalogs based on halo catalogs. 
# Currently support only AbacusSummit simulation. 
#
# Usage  : python3 galaxy_catalog_generator.py [OPTIONS]
# Options: See main setion at the end or use --help option.
#   

__version__ = "0.1a"

import numpy as np
import numpy.ctypeslib as npct, ctypes as ct
import os, os.path, glob, re, json, logging, threading, time

# Loading the shared library and setting up the functions
lib = npct.load_library("libhaloutils", os.path.dirname(os.path.dirname(__file__)))
lib.generate_galaxy_catalog.argtypes = [ct.c_int64, ct.c_int64, ct.c_int, ct.POINTER(ct.c_int)]
lib.generate_galaxy_catalog.restype  = None

def _share_data(
        id            : int,                             # file sharing ID
        hmargs        : dict,                            # halo model parameters      
        bounding_box  : tuple[list[float], list[float]], # simulation bounding box       
        variance_vars : tuple[float, float, int, int],   # (lnma, lnmb, size, filter)
        pktable       : np.ndarray,                      # power spectrum table: lnk vs lnp
        data_generator: map,                             # halo data
        metadata      : dict = {} , 
    ) -> None:
    # Save the arguments to a binary file for sharing. 

    shared_file = f"{id}.vars.dat"
    with open(shared_file, "wb") as fp:
        # First block is the halo model parameters as struct `hmargs_t`...
        hmargs_t = [
            ( "lnm_min"   ,  "<f8"  ),( "sigma_m"   ,  "<f8"  ),
            ( "lnm0"      ,  "<f8"  ),( "lnm1"      ,  "<f8"  ),
            ( "alpha"     ,  "<f8"  ),( "scale_shmf",  "<f8"  ),
            ( "slope_shmf",  "<f8"  ),( "z"         ,  "<f8"  ),
            ( "H0"        ,  "<f8"  ),( "Om0"       ,  "<f8"  ),
            ( "Delta_m"   ,  "<f8"  ),( "dplus"     ,  "<f8"  ),
        ]
        np.array(
            tuple( hmargs.get(field) for field, _ in hmargs_t ), 
            dtype = hmargs_t
        ).tofile(fp)

        # Next block stores the bounding box in Mpc, size of power spectrum table, 
        # filter function code, size and mass range for variance table...  
        lnma, lnmb, table_size, filter_id  = variance_vars
        np.array(
            ( bounding_box, pktable.shape[0], filter_id, table_size, lnma, lnmb ),
            dtype = [("bbox", "<f8", (2, 3)), ("pktab_size", "<i8"), ("filt", "<i4"), 
                     ("ns"  , "<i8"        ), ("lnma"      , "<f8"), ("lnmb", "<f8"),]
        ).tofile(fp)

        # Last block is the power spectrum table.
        pktable.astype("<f8").tofile(fp)
    
    # Metadata is saved in JSON format 
    shared_file = f"{id}.meta.dat"  
    with open(shared_file, 'w') as fp: 
        json.dump(metadata, fp, sort_keys = False, separators = (',', ':'))

    # Save the halo catalog data a binary file for sharing. Passing the halo catalog 
    # data loder as a generator, so that only the needed the data is loaded, making
    # effiicient use of memory. 
    shared_file = f"{id}.hbuf.dat"
    with open(shared_file, 'wb') as fp: 

        # Data written as a stream of `halodata_t` structs.
        halodata_t = [("id", "<i8"), ("pos", "<f8", 3), ("mass", "<f8")]
        for halo_id, halo_pos, halo_mass in data_generator:
            n_halos     = np.size(halo_id)
            halo_buffer = np.empty((n_halos, ), dtype = halodata_t)
            halo_buffer["id"  ] = np.array(halo_id  ).astype("<i8", copy = False)
            halo_buffer["pos" ] = np.array(halo_pos ).astype("<f8", copy = False)
            halo_buffer["mass"] = np.array(halo_mass).astype("<f8", copy = False)
            halo_buffer.tofile(fp)

    return

def _prepare_abacus_workspace(
        id       : int,       # ID for file sharing 
        args     : dict,      # Values for halo model and other parameters
        simname  : str,       # Name of the simulation
        redshift : float,     # Redshift 
        loc      : str = ".", # Path to look for halo catalog files
    ) -> int:
    # Write shared data for galaxy generation using halos and parameters from 
    # AbacusSummit simulation. This will return 0 for success and non-zero on 
    # error.
    
    from math import exp
    from numpy.lib.recfunctions import structured_to_unstructured
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    from abacusnbody.metadata import get_meta

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
        return -1
    except Exception:
        logger.exception(f"error getting metadata for simulation {simname!r} at z={redshift}")
    
    logger.info(f"using power spectrum and growth factor available with the metadata...") 
    
    # Linear power spectrum in the metadata is generated by CLASS at a specific 
    # redshift (`ZD_Pk_file_redshift` in the metadata). This data is stored as an 
    # astropy Table as k (h/Mpc) and P (Mpc/h)^3. 
    h          = meta["H0"] / 100. # Hubble parameter in 100 km/s/Mpc
    pktab      = structured_to_unstructured(np.array(meta["CLASS_power_spectrum"])) # at z_pk
    pktab[:,0] = np.log(pktab[:,0]) + np.log(h)   # k in h/Mpc     -> log(k in 1/Mpc)
    pktab[:,1] = np.log(pktab[:,1]) - np.log(h)*3 # P in (Mpc/h)^3 -> log(P in Mpc^3)

    # Growth factor is calculated using the GrowthTable in the metadata, and
    # power spectrum is interpolated using this value. 
    z_target   = meta["Redshift"]
    z_pk       = meta['ZD_Pk_file_redshift'] 
    dplus_tab  = meta['GrowthTable']
    dplus_at_z = dplus_tab[z_target] / dplus_tab[z_pk] # growth factor at z, w.r.to z_pk
    pktab[:,1] = pktab[:,1] + 2*np.log(dplus_at_z)     # interpolate power spectrum to z
    dplus_at_z = dplus_tab[z_target] / dplus_tab[0.0]  # growth factor at z, w.r.to 0

    # Halo model arguments 
    hmargs = {
        "lnm_min"   :  args["lnm_min"   ]   , 
        "sigma_m"   :  args["sigma_m"   ]   , 
        "lnm0"      :  args["lnm0"      ]   , 
        "lnm1"      :  args["lnm1"      ]   ,
        "alpha"     :  args["alpha"     ]   , 
        "scale_shmf":  args["scale_shmf"]   , 
        "slope_shmf":  args["slope_shmf"]   , 
        "z"         :  meta["Redshift"  ]   ,
        "H0"        :  meta["H0"        ]   , 
        "Om0"       :  meta["Omega_M"   ]   , 
        "Delta_m"   :  meta["SODensity" ][0], 
        "dplus"     :  dplus_at_z           ,
    }

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
        "powerSpectrumTable": pktab.tolist(), # saving the powerspectrum also...
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
        return -1
    
    # Load halo data from the files and add to the shared catalog file.
    logger.info(f"found {len(files)} files for simulation {simname!r} at z={redshift}")

    def load_abacus_halo_catalog(fn: str):
        logger.info(f"loading halo catalog from file: {fn!r}")
        catalog = CompaSOHaloCatalog(fn, cleaned = False, fields = ["id", "SO_central_particle", "N"])

        h = catalog.header["H0"] / 100. 
        unit_mass_Msun = catalog.header["ParticleMassMsun"] 
        
        uid  = catalog.halos["id"]                      # halo ID  
        pos  = catalog.halos["SO_central_particle"] / h # halo position coordinates in Mpc
        mass = catalog.halos["N"] * unit_mass_Msun      # halo mass in Msun
        return uid, pos, mass
    
    # Save the values as shared data: 
    _share_data(
        id, 
        hmargs, 
        bounding_box_Mpc, 
        [lnma, lnmb, table_size, filter_id], 
        pktab, 
        map(load_abacus_halo_catalog, files), 
        meta, 
    )
    return 0

def _generate_galaxies(id: int, nthreads: int = -1, rseed: int = None) -> None:
    # Generate galaxies using shared halo catalog and parameters. Call this only  
    # after creating the shared files.  
    
    assert os.path.exists( f"{id}.vars.dat" ), "missing shared variable data"
    assert os.path.exists( f"{id}.hbuf.dat" ), "missing shared halo catalog data"

    if nthreads < 1 : nthreads = os.cpu_count()
    if rseed is None: rseed    = int( time.time() )    

    stop_event = threading.Event()
    
    # Start a thread for logging
    thread = threading.Thread(target = _log_watcher, args = (id, stop_event), daemon = True)
    thread.start()
    
    # Galaxy generation
    error_flag = ct.c_int(1) # Error code: non-zero on error 
    lib.generate_galaxy_catalog(id, rseed, nthreads, ct.byref(error_flag))

    if error_flag.value != 0:
        logger = logging.getLogger()
        logger.error(f"galaxy catalog generation failed with code {error_flag.value}")
        stop_event.set() # setting the stop event to end the thread
        thread.join()
        return -1

    # Wait for logging thread to finish
    thread.join()

    return

def _log_watcher(id: int, stop_event: threading.Event) -> None:
    # Watch the log file, pass the lines to main log file as they appear.

    logger   = logging.getLogger()
    log_file = f"{id}.log"  # Filename for log file associated with this process
    one_day  = 86400        # Seconds in a day

    # Wait until the file is created
    start_time = time.time()
    while not os.path.exists(log_file):
        if stop_event.is_set(): return
        if time.time() - start_time > one_day: return # set a wait timeout of 1 day
        time.sleep(0.1)

    with open(log_file, "r") as f:
        start_time = time.time()
        while not stop_event.is_set():
            message = f.readline().strip()
            if not message: 
                if time.time() - start_time > one_day: return # set a wait timeout of 1 day
                time.sleep(0.1) # wait until next line is get written
                continue
            if message == "END": break # end of the file is marked by the sentinal 'END'
            logger.info(message)       # write the log message 
            start_time = time.time()   # reset the wait time, if message is recieved

    # Since the messages in the process specific log file are written to 
    # the main log file, that file is non longer required - delelting it...
    try:
        logger.info(f"deleting {log_file!r}...")
        os.remove( log_file )
        # Exceptions are ignored, with printing a log message for that.
    except PermissionError:
        logger.error("failed to delete log file - premission denied")
    except Exception:
        logger.exception("error deleting log file.")  
    return 

def _save_data_as_asdf(id: int, path: str = '.') -> None:
    # Save data from the shared output to ASDF file(s) in the given path. 

    import asdf

    logger = logging.getLogger()

    # Loading metadata, if any. Sonce this file is not needed, after that it is 
    # deleted. 
    meta = {}
    shared_meta_file = f"{id}.meta.dat"
    if os.path.exists(shared_meta_file):
        with open(shared_meta_file, 'r') as fp:
            meta = json.load(fp)

    # Folder for saving output ASDF files (create if not exist)
    outdir = os.path.join( os.path.abspath(os.path.expanduser(path)), "galaxy_info" )
    os.makedirs(outdir, exist_ok = True)
    logger.info(f"saving galaxy catalog files to {outdir!r}...")

    # Writing a seperate file containing all items in the metadata: that is,
    # ndarrays like power spectrum table are stored in specific format in 
    # this file. These data are then removed from metadata section that is
    # in the catalog files.
    fields_to_delete = []
    if "powerSpectrumTable" in meta: 
        meta["powerSpectrumTable"] = np.array( 
            [ (_lnk_, _lnp_) for _lnk_, _lnp_ in meta["powerSpectrumTable"] ], 
            dtype = [("lnk", "<f8"), ("lnp", "<f8")],
        )
        fields_to_delete.append( "powerSpectrumTable" )

    with asdf.AsdfFile(meta) as af: 
        fn = os.path.join(outdir, "metadata.asdf")
        af.write_to(fn, all_array_compression = 'zlib')
    
    for field in fields_to_delete: _ = meta.pop(field, None) # delete fields
    
    # Loading galaxy data from the binary file and save them in ASDF files of maximum 
    # 1 GiB size. Each file will store the metadata (header), and the data section 
    # contains position (Mpc), mass (Msun), parent halo ID and galaxy type (c for 
    # central, s for satellite).     
    shared_galaxy_file = f"{id}.gbuf.dat"
    galaxy_file_size   = os.path.getsize(shared_galaxy_file) # size of the output file in bytes
    record_size        = 41 # size of galaxt catalog record in bytes
    assert galaxy_file_size % record_size == 0, "galaxy catalog filesize should be multiple of 41"

    total_items        = galaxy_file_size // record_size # total numebr of records
    max_items_per_file = 1073741824 // record_size       # maximum number of records (filesize: 1 GiB)
    if total_items % max_items_per_file > 0:
        max_items_per_file += 1

    with open(shared_galaxy_file, 'r') as fp:
        i = 0
        while True:
            galaxy_buffer = np.fromfile(
                fp, 
                dtype = [("id", "<i8"), ("pos", "<f8", 3), ("mass", "<f8"), ("typ", "S1")], 
                count = max_items_per_file,
            )
            if galaxy_buffer.shape[0] < 1: break
            
            with asdf.AsdfFile({
                "header" : meta, 
                "data"   : {
                    "parentHaloID"   : galaxy_buffer["id"  ], 
                    "galaxyPosition" : galaxy_buffer["pos" ], 
                    "galaxyMass"     : galaxy_buffer["mass"], 
                    "galaxyType"     : galaxy_buffer["typ" ],
                } 
            }) as af:
                fn = os.path.join(outdir, f"galaxy_info_{i:03d}.asdf")
                af.write_to(fn, all_array_compression = 'zlib')
                i += 1
        logger.info(f"written {i} galaxy catalog files.")

    return

def _clean_up(id: int) -> None:
    #  Clean up unwanted files, optionally keep halo catalog.

    logger = logging.getLogger()

    files_to_delete = [ f"{id}.meta.dat", f"{id}.vars.dat", f"{id}.gbuf.dat", f"{id}.hbuf.dat" ]
    for file in files_to_delete:
        if not os.path.exists(file): continue # file already deleted
        try:
            logger.info(f"deleting {file!r}...")
            os.remove( file )
            # Exceptions are ignored, with printing a log message for that.
        except PermissionError:
            logger.error(f"failed to delete file {file!r} - premission denied")
        except Exception:
            logger.exception(f"error deleting file {file!r}.")

    return

from typing import Literal

def galaxy_catalog_generator(
        simname      : str,          # Name of the simulation 
        redshift     : float,        # Redshift value 
        mmin         : float,        # Threshold mass for central galaxy formation, Mmin (Msun)  
        m0           : float,        # Threshold mass for satellite galaxy formation, M0 (Msun)
        m1           : float,        # Amplitude parameter for satellite count, M1 (Msun)   
        sigma_m      : float =  0. , # Width parameter for central count, sigmaM (0=step function)
        alpha        : float =  1. , # Index parameter for satellite count, alpha 
        scale_shmf   : float =  0.5, # Scale parameter for subhalo mass-function 
        slope_shmf   : float =  2. , # Slope parameter for subhalo mass-function  
        filter_fn    : Literal["tophat", "gauss"] = "tophat", # Filter function
        sigma_size   : int   = 101,  # Size of the variance table
        output_path  : str   = '.',  # Path to save output files 
        catalog_path : str   = '.',  # Path to search halo catalogs (simulation specific tree)
        nthreads     : int   = -1 ,  # Number of threads (default=cpu_count) 
        rseed        : int   = None, # Random seed value (RSEED)
        process_id   : int   = None, # ID for data sharing (FID) 
    ) -> None:
    r"""
    Generate galaxy catalogs based on a halo catalog.
    """

    from math import log
    
    # Value of process_id is used for data sharing. If not specified, process ID
    # of this process is used. Its value can also passed using the environment
    # variable FID. It should be a positive integer. In case of incorrect value, 
    # default is used. 
    if not isinstance(process_id, int) or process_id < 0:
        process_id = os.environ.get("FID", os.getpid())
    try:
        process_id = int( process_id )
    except ValueError:
        process_id = os.getpid()  

    # Random seed value can also be passed as an environment variable. It should
    # be a positive integer. Otherwise, a default value is generated based on the
    # current time. 
    if not isinstance(rseed, int) or rseed < 0:
        rseed = os.environ.get("RSEED", "None")
        rseed = int(rseed) if str.isnumeric(rseed) else None
            
    # Setting up shared data based on simulation. Type of simulation is calculated
    # based on the prefix of its name.  
    if simname.startswith("AbacusSummit"):
        error_code = _prepare_abacus_workspace(
            process_id, 
            {
                "lnm_min"   : log(mmin),
                "sigma_m"   : sigma_m,
                "lnm0"      : log(m0),
                "lnm1"      : log(m1),
                "alpha"     : alpha,
                "scale_shmf": scale_shmf,
                "slope_shmf": slope_shmf,
                "sigma_size": sigma_size,
                "filter"    : filter_fn,
            }, 
            simname, 
            redshift, 
            catalog_path,
        )
    else:
        raise RuntimeError(f"simulation {simname!r} is not supported")
    if error_code != 0:
        return # catalog generation failed :(
    
    # Generating the galaxies
    _generate_galaxies(process_id, nthreads, rseed)

    # Saving data
    _save_data_as_asdf(process_id, output_path)

    # Cleaning up working folder
    _clean_up(process_id)
    
    return

if __name__ == "__main__":

    import click, logging.config, warnings
    from click import Choice, IntRange, Path

    warnings.catch_warnings(action = "ignore")
    
    def get_log_filename(): 
        p  = os.path.join( os.getcwd() , "logs" )
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

    options = [
        (["--simname"     ], str                        , "Name of simulation"              ,            ),
        (["--redshift"    ], float                      , "Redshift value"                  ,            ),
        (["--mmin"        ], float                      , "Halo model parameter Mmin (Msun)",            ),
        (["--m0"          ], float                      , "Halo model parameter M0 (Msun)"  ,            ),
        (["--m1"          ], float                      , "Halo model parameter M1 (Msun)"  ,            ),
        (["--sigma-m"     ], float                      , "Halo model parameter sigmaM"     ,  0.        ),
        (["--alpha"       ], float                      , "Halo model parameter alpha"      ,  1.        ),
        (["--scale_shmf"  ], float                      , "Scale parameter for SHMF"        ,  0.5       ),
        (["--slope_shmf"  ], float                      , "Slope parameter for SHMF"        ,  2.        ),
        (["--filter_fn"   ], Choice(["tophat", "gauss"]), "Filter function for variance"    , "tophat"   ),
        (["--sigma_size"  ], IntRange(3)                , "Size of variance table"          , 101        ),
        (["--output_path" ], Path(file_okay = False)    , "Path to write output files"      , os.getcwd()),
        (["--catalog_path"], Path(exists = True )       , "Path to look for catalog files"  , os.getcwd()),
        (["--nthreads"    ], int                        , "Number of threads to use"        , -1         ),
    ]
    cli = galaxy_catalog_generator
    for options, otype, help, *default in reversed(options):
        if not default:
            cli = click.option(*options, type = otype, required = True, help = help)(cli)
        else:
            cli = click.option(*options, type = otype, default = default[0], help = help)(cli)
    cli = click.version_option(__version__, message = "%(prog)s v%(version)s")(cli) 
    cli = click.command(cli)
    cli()

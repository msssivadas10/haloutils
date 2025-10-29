
__version__ = "0.6a"

import os, logging, asdf, multiprocessing as mp, numpy as np
from pathlib import Path
from collections import namedtuple
from typing import Callable, Any
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

pcresult = namedtuple("pcresult" , ["D1D2", "R1R2", "D1R2", "D2R1", "ND1", "ND2", "NR1", "NR2"])
def pair_count_3d(
        D1       : Any, # all coords must be in the range [-boxsize/2, boxsize/2]...
        D2       : Any, # None, for using D2 = D1
        R1       : Any,
        R2       : Any, # None, for using R2 = R1
        rbins    : list[float],
        boxsize  : float, 
        periodic : bool  = False,
        subdivs  : int   =  3,
        nthreads : int   = -1,
    ) -> tuple[pcresult, list[pcresult]]:
    # Calculate the 3D pair counts between D1, D2 and R1, R2 required for correlation function
    # calculations. Return the pair counts for the full region and jacknife counts (by excluding 
    # sub-regions one at a time).   

    from math import ceil
    from operator import sub
    from itertools import product
    from Corrfunc.theory.DD import DD
    
    # STEP 1: Listing all the possible sub-box pairs that can contribute to the total
    # pair counts. Each sub-box is represented by its flat index. 

    subboxsize = boxsize / subdivs
    near_boxes = ceil(max(rbins) / subboxsize) # no. of boxes near to this box on each face
    box_pairs  = []
    for ix, iy, iz in product( range(subdivs), repeat = 3 ):
        i = ix*subdivs**2 + iy*subdivs + iz # flat index
        for dx, dy, dz in product( range(-near_boxes, near_boxes+1), repeat = 3 ):
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            if periodic: # apply periodic wrapping
                jx, jy, jz = jx % subdivs, jy % subdivs, jz % subdivs
            elif any( q < 0 or q >= subdivs for q in (jx, jy, jz) ): # cell outside total box
                continue
            j = jx*subdivs**2 + jy*subdivs + jz # flat index
            box_pairs.append((i, j))
    
    # STEP 2: Sorting the points based on their sub-box index. This will result in the
    # points corresponding to same box grouped together.  

    def sort_data_by_subbox(data, boxsize, subdivs):
        box = np.floor((data + boxsize / 2) / (boxsize / subdivs)).astype("i8")  
        box = np.clip(box, 0, subdivs-1) # set points outside to sub-regions at edges 
        box = box[:,0]*subdivs**2 + box[:,1]*subdivs + box[:,2] # flat index

        sorted_order  = np.argsort(box)
        box, data[:]  = box[sorted_order], data[sorted_order, :]        
        box_id, start = np.unique(box, return_index = True)
        box_info      = {
            k: ( start[i], start[i+1] if i+1 < len(start) else len(box) )
                for i, k in enumerate(box_id)
        }
        return data, box_info
    
    D2_is_D1 = D2 is None or D2 is D1
    R2_is_R1 = R2 is None or R2 is R1

    D1, D1_box_info = sort_data_by_subbox(D1, boxsize, subdivs)
    R1, R1_box_info = sort_data_by_subbox(R1, boxsize, subdivs)
    D2, D2_box_info = sort_data_by_subbox(D2, boxsize, subdivs) if not D2_is_D1 else (D1, D1_box_info)
    R2, R2_box_info = sort_data_by_subbox(R2, boxsize, subdivs) if not R2_is_R1 else (R1, R1_box_info)

    # STEP 3: Pair counting. It is done between possible sub-box pairs and later combined
    # to get total jackknife results.  

    rbins = np.asarray(rbins, dtype = "f8")
    if nthreads < 1: nthreads = os.cpu_count()

    # Counts between D1 and D2:
    DD_counts = {}
    for i, j in box_pairs:
        if D2_is_D1 and j < i: continue
        if i in D1_box_info and j in D2_box_info:
            i_start, i_stop = D1_box_info[i]
            j_start, j_stop = D2_box_info[j]
            Di, Dj          = D1[i_start:i_stop, :], D2[j_start:j_stop, :]
            count_results   = DD(
                X1 = Di[:, 0], Y1 = Di[:, 1], Z1 = Di[:, 2], 
                X2 = Dj[:, 0], Y2 = Dj[:, 1], Z2 = Dj[:, 2], 
                autocorr = ( i == j and D2_is_D1 ), 
                nthreads = nthreads, 
                binfile  = rbins, 
                periodic = periodic, 
                boxsize  = boxsize, 
                verbose  = False,
            )
            DD_counts[(i, j)] = count_results["npairs"]
            if D2_is_D1 and j != i: 
                DD_counts[(j, i)] = DD_counts[(i, j)]
    
    # Counts between R1 and R2:
    RR_counts = {}
    for i, j in box_pairs:
        if R2_is_R1 and j < i: continue
        if i in R1_box_info and j in R2_box_info:
            i_start, i_stop = R1_box_info[i]
            j_start, j_stop = R2_box_info[j]
            Ri, Rj          = R1[i_start:i_stop, :], R2[j_start:j_stop, :]
            count_results   = DD(
                X1 = Ri[:, 0], Y1 = Ri[:, 1], Z1 = Ri[:, 2], 
                X2 = Rj[:, 0], Y2 = Rj[:, 1], Z2 = Rj[:, 2], 
                autocorr = ( i == j and R2_is_R1 ), 
                nthreads = nthreads, 
                binfile  = rbins, 
                periodic = periodic, 
                boxsize  = boxsize, 
                verbose  = False,
            )
            RR_counts[(i, j)] = count_results["npairs"] 
            if R2_is_R1 and j != i:
                RR_counts[(j, i)] = RR_counts[(i, j)]

    # Counts between D1 and R2, D2 and R1:
    DR_counts, RD_counts = {}, {}
    for i, j in box_pairs:
        if i in D1_box_info and j in R2_box_info:
            i_start, i_stop = D1_box_info[i]
            j_start, j_stop = R2_box_info[j]
            Di, Rj          = D1[i_start:i_stop, :], R2[j_start:j_stop, :]
            count_results   = DD(
                X1 = Di[:, 0], Y1 = Di[:, 1], Z1 = Di[:, 2], 
                X2 = Rj[:, 0], Y2 = Rj[:, 1], Z2 = Rj[:, 2], 
                autocorr = 0, 
                nthreads = nthreads, 
                binfile  = rbins, 
                periodic = periodic, 
                boxsize  = boxsize, 
                verbose  = False,
            )
            DR_counts[(i, j)] = count_results["npairs"] 
        
        if D2_is_D1 and R2_is_R1: 
            # If D1 and R1 same as D2 and R2, then D1R2 will be same as D2R1, so using 
            # the already computed counts... 
            RD_counts[(i, j)] = DR_counts[(i, j)]
        elif i in D2_box_info and j in R1_box_info:
            i_start, i_stop = D2_box_info[i]
            j_start, j_stop = R1_box_info[j]
            Di, Rj          = D2[i_start:i_stop, :], R1[j_start:j_stop, :]
            count_results   = DD(
                X1 = Di[:, 0], Y1 = Di[:, 1], Z1 = Di[:, 2], 
                X2 = Rj[:, 0], Y2 = Rj[:, 1], Z2 = Rj[:, 2], 
                autocorr = 0, 
                nthreads = nthreads, 
                binfile  = rbins, 
                periodic = periodic, 
                boxsize  = boxsize, 
                verbose  = False,
            )
            RD_counts[(i, j)] = count_results["npairs"] 

    # STEP 4: Combining the results

    # Total counts are calculated by summing contributions from all sub-boxes...
    full_result = pcresult(
        D1D2 = sum( DD_counts.values() ),
        R1R2 = sum( RR_counts.values() ),
        D1R2 = sum( DR_counts.values() ),
        D2R1 = sum( RD_counts.values() ),
        ND1  = len(D1),
        ND2  = len(D2),
        NR1  = len(R1),
        NR2  = len(R2),
    )

    # Jackknife count corresponding to sub-box is calculated by summing contributions
    # from all sub-boxes, except those connected to points in that sub-box... 
    jack_results = [
        pcresult(
            D1D2 = full_result.D1D2 - sum([ arr for ij, arr in DD_counts.items() if k in ij ]),
            R1R2 = full_result.R1R2 - sum([ arr for ij, arr in RR_counts.items() if k in ij ]),
            D1R2 = full_result.D1R2 - sum([ arr for ij, arr in DR_counts.items() if k in ij ]),
            D2R1 = full_result.D2R1 - sum([ arr for ij, arr in RD_counts.items() if k in ij ]),
            ND1  = full_result.ND1  + sub( *D1_box_info.get( k, (0, 0) ) ), # total + (start - stop)
            ND2  = full_result.ND2  + sub( *D2_box_info.get( k, (0, 0) ) ),
            NR1  = full_result.NR1  + sub( *R1_box_info.get( k, (0, 0) ) ), 
            NR2  = full_result.NR2  + sub( *R2_box_info.get( k, (0, 0) ) ), 
        )
        for k in range( subdivs**3 )
    ]
    
    return full_result, jack_results

def correlation_from_count(pc: pcresult, estimator: str = 'ls') -> Any:
    # Calculate the correlation function from given 3D pair counts.  

    def _float(x): 
        try: 
            return float(x)
        except TypeError: 
            return np.asarray(x, dtype = float)
        
    def normalize_count(cnt, n1, n2):
        n1n2 = _float(n1) * _float(n2)
        try:
            return np.true_divide( cnt, n1n2 )
        except ValueError:
            return np.true_divide( cnt, np.reshape(n1n2, (-1, 1)) )

    pairs = { # normalized pair counts
        k: normalize_count( 
                getattr(pc,  s1+s2), 
                getattr(pc, 'N'+s1), 
                getattr(pc, 'N'+s2),
            ) 
            for s1, s2, k in [ ("D1","D2","DD"), ("R1","R2","RR"), ("D1","R2","DR"), ("D2","R1","RD") ]
            if getattr(pc, s1+s2) is not None
    } 
    if "DR" in pairs and "RD" in pairs: 
        pairs["DR"] = ( pairs["DR"] + pairs.pop("RD") ) / 2.
    elif "RD" in pairs: pairs["DR"] = pairs.pop("RD")

    if estimator.lower() in [ "peebles-hauser", "natural" ]: 
        if "RR" not in pairs: raise ValueError("missing RR counts")
        xi    = np.zeros_like( pairs["DD"] )
        xi[:] = np.nan
        m     = ( pairs["RR"] != 0. )
        xi[m] = pairs["DD"][m] / pairs["RR"][m] - 1
        return xi
    
    if estimator.lower() in [ "davis-peebles" , "dp" ]: 
        if "DR" not in pairs: raise ValueError("missing DR counts")
        xi    = np.zeros_like( pairs["DD"] )
        xi[:] = np.nan
        m     = ( pairs["DR"] != 0. )
        xi[m] = pairs["DD"][m] / pairs["DR"][m] - 1
        return xi
    
    if estimator.lower() in [ "hamilton", "ham" ]: 
        if "RR" not in pairs: raise ValueError("missing RR counts")
        if "DR" not in pairs: raise ValueError("missing DR counts")
        xi    = np.zeros_like( pairs["DD"] )
        xi[:] = np.nan
        m     = ( pairs["DR"] != 0. )
        xi[m] = (pairs["DD"][m] * pairs["RR"][m]) / (pairs["DR"][m]**2) - 1
        return xi
    
    if estimator.lower() in [ "landy-szalay", "ls" ]: 
        if "RR" not in pairs: raise ValueError("missing RR counts")
        if "DR" not in pairs: raise ValueError("missing DR counts")
        xi    = np.zeros_like( pairs["DD"] )
        xi[:] = np.nan
        m     = ( pairs["RR"] != 0. )
        xi[m] = (pairs["DD"][m] - 2*pairs["DR"][m] + pairs["RR"][m]) / pairs["RR"][m]
        return xi

    raise ValueError(f"unknown estimator {estimator!r}")

# Pair counting for specific catalogs:

def _cleanup(fn):

    import inspect, functools, tempfile, shutil

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

_CORRFUNC_SUCCESS = 0
_CORRFUNC_MISSING_CATALOGS = -1

@_cleanup
def _corrfunc(
        loader       : Callable[[str, tuple[float, float]], Any],
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
        pktab        : Any, 
        saved_rbins  : list[float] = None,
    ) -> int:

    from os.path import getsize
    from base64 import b64encode
    from struct import pack
    from numpy.random import default_rng

    def load_catalog(fn: Path, save_catalog=None, *args):

        # Load and save the catalog as a memory mappable file, if not exist:
        if not fn.exists(): save_catalog(fn, *args)
            
        # Getting the size of the saved catalog from the file: size (in bytes) must be a 
        # multiple of 3*sizeof(float64)... 
        FLOAT64_SIZE = 8 # No. of bytes for float64
        n_pts, __rem_bytes = divmod( getsize(fn), (3*FLOAT64_SIZE) )
        assert __rem_bytes == 0

        if n_pts == 0: 
            return np.empty((0, 3), dtype = "f8") 

        # Loading the catalog as a memmap:
        arr = np.memmap(fn, dtype = "f8", mode = 'r+', shape = (n_pts, 3)) 
        return arr
    
    def save_catalog(fn, loader, files, mrange, nthreads):
        # Load catalog and save as a memory-mappable file 
        with open(fn, 'wb') as fp, mp.Pool(processes = nthreads) as pool:
            for data in pool.starmap( loader, [ (file, mrange) for file in files ] ):
                np.array(data, copy = None).astype("f8").tofile(fp)
        return
    
    def save_random(fn, boxsize, rseed, randsize, *catsize):
        # Save random catalog as a memory-mappable file 
        rng    = default_rng(rseed)
        n_rand = int( randsize*max(catsize) ) # size of the random catalog
        rng.uniform( -boxsize / 2., boxsize / 2. , size = (n_rand, 3) ).astype("f8").tofile(fn)
        return

    def encode_mrange(x): 
        return b64encode( pack("<dd", *x), altchars = b"+-" ).decode("utf-8")

    logger  = logging.getLogger()
    workdir = Path(workdir).expanduser().absolute()
    if nthreads < 1: nthreads = os.cpu_count() 

    # Setting up pair counting: if the catalogs are available in the current directory use 
    # them, otherwise load from the halo catalog files.
    fn1 = workdir.joinpath(fn_prefix + encode_mrange( mrange1 or (0., np.inf) ) + ".bin") # filename for dataset #1
    fn2 = workdir.joinpath(fn_prefix + encode_mrange( mrange2 or (0., np.inf) ) + ".bin") # filename for dataset #2 
    if not fn1.exists() or not fn2.exists(): 
        files = list( Path(loc).glob( file_pattern ) )
        if not files: return _CORRFUNC_MISSING_CATALOGS
    else: 
        files = []
    
    # Loading the data as memory maps:    
    logger.info("getting data for mass range {0[0]:.3e}..{0[1]:.3e}...".format(mrange1 or (0., np.inf)))
    d1 = load_catalog( fn1, save_catalog, loader, files, mrange1, nthreads )  

    logger.info("getting data for mass range {0[0]:.3e}..{0[1]:.3e}...".format(mrange2 or (0., np.inf)))
    d2 = load_catalog( fn2, save_catalog, loader, files, mrange2, nthreads ) 

    # Generating random catalog. NOTE: Since both sets are from the same catalog, 
    # no second random catalog is needed. 
    logger.info(f"generating random catalog...")
    fnr = workdir.joinpath("corrfunc.rand.bin")
    r1  = load_catalog(fnr, save_random, boxsize, rseed, randsize, len(d1), len(d2))

    logger.info("completed setting-up for correlation.")

    # Count number of pairs between the datasets: this will count the no. of pairs 
    # for full datasets, and jackknife subdivisions of the entire box (if enabled). 
    logger.info(f"calculating pair counts with ND1={len(d1)}, ND2={len(d2)}, NR={len(r1)}"\
                f" and {subdivs**3} jacknife subregions")
    full_result, jack_results = pair_count_3d(
        d1, d2, r1, None, 
        rbins    = rbins, 
        boxsize  = boxsize, 
        periodic = True, 
        subdivs  = subdivs,
        nthreads = nthreads
    )
    meta.update({
        "jackknifeSamples": len(jack_results),
        "mrange1"         : mrange1,
        "mrange2"         : mrange2,
    })
    try: os.remove(fnr) # delete random catalog
    except: pass
    
    # Saving data:
    logger.info(f"saving data to {str(outfile)!r}")
    with asdf.AsdfFile({
        "header": meta,
        "data"  : {
            "powerSpectrum"   : pktab,
            "rbins"           : rbins if saved_rbins is None else saved_rbins, 
            "pairCounts_full" : full_result._asdict(),
            "pairCounts_jack" : {
                _field: np.stack([ getattr(t, _field) for t in jack_results ], axis = 0) 
                    for _field in pcresult._fields
            }, 
        }
    }) as af:
        af.write_to(
            str( Path(outfile).expanduser().absolute() ), 
            all_array_compression = 'zlib', 
        )
    return _CORRFUNC_SUCCESS

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
    # Calculate pair counts between abacus halo catalogs (with different mass ranges).

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
    
    rbins  = np.array(rbins, dtype = "f8") # value in Mpc
    dirs   = [  simname, "halos", f"z{redshift:.3f}", "halo_info"  ]
    retval = _corrfunc(
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
    )
    if retval == _CORRFUNC_MISSING_CATALOGS:
        raise FileNotFoundError(f"missing files in {str(loc)!r} for {simname!r}, z={redshift}")
    return

def _load_abacus(file: str, mrange: tuple[float, float]) -> Any:
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
    # Calculate pair counts between galaxy catalogs (with different parent halo mass ranges).

    loc   = Path(loc).expanduser().absolute().joinpath("galaxy_info")
    files = list( loc.glob("galaxy_info_*.asdf") )
    if not files: 
        raise FileNotFoundError(f"missing catalog files in {str(loc)!r}")
    
    # Loading metadata
    with asdf.open(files[0]) as af: meta = af["header"]
    pkfile = loc.joinpath('powerspectrum.txt')
    pktab  = np.loadtxt(pkfile) if pkfile.exists() else []

    rbins  = np.array(rbins, dtype = "f8") # value in Mpc
    retval = _corrfunc(
        loader           = _load_galaxies, 
        fn_prefix        = "galaxy.", 
        file_pattern     = "galaxy_info_*.asdf", 
        mrange1          =  mrange1, 
        mrange2          =  mrange2, 
        boxsize          =  meta["boxsizeMpc"], # value in Mpc
        loc              =  loc, 
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
    )
    return

def _load_galaxies(file: str, mrange: tuple[float, float]) -> Any:
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

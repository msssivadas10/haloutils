
import numpy as np, numpy.random as rand, numpy.typing as nt 
import os, os.path, tempfile, logging
from collections import namedtuple

sizes_t      = namedtuple("sizes_t",      ["d1", "d2", "r1", "r2"])
paircounts_t = namedtuple("paircounts_t", ["sizes", "d1d2", "d1r2", "d2r1", "r1r2"])

def _count_pairs(
        data1    : nt.NDArray[np.float64],
        data2    : nt.NDArray[np.float64],
        rand1    : nt.NDArray[np.float64],
        rand2    : nt.NDArray[np.float64],
        r        : list[float], 
        boxsize  : float, 
        periodic : bool  = True,
        nthreads : int   = -1, 
    ) -> paircounts_t:
    
    from Corrfunc.theory.DD import DD

    def DD_wrapper(d1, d2=None, **kwargs):
        data = { f"{c}1": d1[:,i] for i, c in enumerate("XYZ") }
        if d2 is not None:
            data.update({ f"{c}2": d2[:,i] for i, c in enumerate("XYZ") })
            autocorr = 0 
        else:
            autocorr = 1
        count_result = DD(autocorr = autocorr, **data, **kwargs)
        return count_result["npairs"]
    
    logger = logging.getLogger()

    settings = dict(
        nthreads = os.cpu_count() if nthreads < 1 else nthreads, 
        binfile  = r, 
        periodic = periodic,
        boxsize  = boxsize, 
        verbose  = False,
    )

    logger.info(f"counting pairs of d1 and r2...")
    d1r2 = DD_wrapper(data1, rand2, **settings)

    if data1 is data2:
        logger.info(f"counting pairs of d1 and r2...")
        d1d2 = DD_wrapper(data1, **settings)
        d2r1 = d1r2
    else:
        logger.info(f"counting pairs of d1 and r2...")
        d1d2 = DD_wrapper(data1, data2, **settings)
        
        logger.info(f"counting pairs of d2 and r1...")
        d2r1 = DD_wrapper(rand1, data2, **settings)
    
    logger.info(f"counting pairs of r1 and r2...")
    if rand2 is rand1:
        r1r2 = DD_wrapper(rand1, None, **settings)
    else:
        r1r2 = DD_wrapper(rand1, rand2, **settings)

    count_result = paircounts_t(
        sizes_t(
            data1.shape[0], 
            data2.shape[0], 
            rand1.shape[0], 
            rand2.shape[0], 
        ), 
        d1d2, 
        d1r2, 
        d2r1, 
        r1r2, 
    )
    return count_result

pcresult_t = namedtuple("pcresult_t", ["r_centers", "full", "jack_samples"])
def count_pairs(
        data1    : nt.NDArray[np.float64],
        data2    : nt.NDArray[np.float64],
        r        : list[float], 
        boxsize  : float, 
        periodic : bool  = True,
        rfrac    : float =  3., 
        nthreads : int   = -1, 
        seed     : int   = None,
        diff_r   : bool  = False,
        ndiv     : int   = 1,
    ) -> pcresult_t:

    from math import floor

    def subdivide_box(data, boxsize, ndiv):

        # Normalize coords to [0, ndiv)
        coords = np.floor( data / (boxsize / ndiv) ).astype("i8")
        coords = np.clip(coords, 0, ndiv-1)
        
        # Flatten (ix,iy,iz) -> j
        sub_indices = (
            coords[:,0] + 
            coords[:,1] * ndiv + 
            coords[:,2] * ndiv**2
        )
        return sub_indices
    
    def make_memmap(tempdir, data, name):
        # Write to binary file
        fname = os.path.join(tempdir, f"{name}.bin")
        data.astype(np.float64).tofile(fname)

        # Memory-map back
        shape = (data.shape[0], data.shape[1])
        arr   = np.memmap(fname, dtype=np.float64, mode='r', shape=shape)
        return arr, fname
    
    logger = logging.getLogger()
    
    data1  = np.array(data1, copy = None)
    nd1, _ = data1.shape 
    if data2 is not None:
        data2  = np.array(data2, copy = None)
        nd2, _ = data2.shape
    else:
        data2, nd2 = data1, nd1
        diff_r     = False

    logger.info("generating random catalogs...")
    rngen  = rand.default_rng(seed)
    if diff_r:
        nr1, nr2 = floor(rfrac*nd1), floor(rfrac*nd2)
        rand1    = rngen.uniform(-boxsize/2., boxsize/2., size = (nr1, 3))
        rand2    = rngen.uniform(-boxsize/2., boxsize/2., size = (nr2, 3)) 
    else:
        nr1   = floor(rfrac*max( nd1, nd2 ))
        nr2   = nr1
        rand1 = rngen.uniform(-boxsize/2., boxsize/2., size = (nr1, 3))
        rand2 = rand1

    # Full-sample correlation (for reference)
    logger.info("counting pairs for full sample...")
    pair_counts_full = _count_pairs(
        data1, data2, rand1, rand2, r, 
        boxsize, 
        periodic, 
        nthreads, 
    )

    # Sub-region correlation (for jackknife error estimate)
    pair_counts_jack = []
    if ndiv > 1: 
        nsubs = ndiv**3 # number of subdivisions of the box

        # Assign sub-box indices
        sub_d1 = subdivide_box( data1, boxsize, ndiv )
        sub_d2 = subdivide_box( data2, boxsize, ndiv )
        sub_r1 = subdivide_box( rand1, boxsize, ndiv )
        sub_r2 = subdivide_box( rand2, boxsize, ndiv )

        with tempfile.TemporaryDirectory() as tmpdir:
            
            for k in range(nsubs):
                # Mask out one subregion:
                fns = [ None ]*4
                _data1, fns[0] = make_memmap( tmpdir, data1[sub_d1 != k], "d1" )
                _data2, fns[1] = make_memmap( tmpdir, data2[sub_d2 != k], "d2" )
                _rand1, fns[2] = make_memmap( tmpdir, rand1[sub_r1 != k], "r1" )
                _rand2, fns[3] = make_memmap( tmpdir, rand2[sub_r2 != k], "r2" )

                logger.info(f"counting pairs for {k} sub-sample...")
                try:
                    counts_j = _count_pairs(
                        _data1, _data2, _rand1, _rand2, r, 
                        boxsize, 
                        periodic, 
                        nthreads, 
                    )
                    pair_counts_jack.append( counts_j )
                finally:
                    for fn in fns:
                        if os.path.exists(fn): os.remove(fn)

    r_centers = np.sqrt( np.multiply(r[1:], r[:-1]) )

    return pcresult_t(r_centers, pair_counts_full, pair_counts_jack)
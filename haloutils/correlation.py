# Pair counting and correlation function calculations.

__version__ = '0.1a'

__all__ = [ 'Grid' ]

import os, os.path, dataclasses
import numpy as np, numpy.typing as npt
from ctypes import c_double, c_int64, c_int
from numpy.ctypeslib import load_library, ndpointer
def _p(dtype, ndim=0): return ndpointer(dtype, ndim, flags="C_CONTIGUOUS")

boxinfo_t = [("boxsize", "f8", 3), ("origin", "f8", 3), ("gridsize", "i8", 3), ("cellsize", "f8", 3)]
cinfo_t   = [("start", "i8"), ("count", "i8")]

# Loading the shared library and setting up the functions
lib = load_library("libhaloutils", os.path.dirname(__file__))
lib.build_grid_hash.argtypes = [
    c_int64  ,  c_int64      , _p(c_double, 2), _p(boxinfo_t),      
    c_int64  , _p(cinfo_t, 1), _p(c_int64 , 1),  c_int, 
    _p(c_int),       
]
lib.count_pairs.argtypes = [
    c_int64,    c_int       ,    c_int      , _p(boxinfo_t, 0),   
    c_int64, _p(c_double, 1), _p(c_int64, 1),    c_int64  ,              
    c_int64, _p(cinfo_t , 1), _p(c_int64, 1), _p(c_double , 2), 
    c_int64, _p(cinfo_t , 1), _p(c_int64, 1), _p(c_double , 2), 
    c_int  ,    c_int       , _p(c_int)     ,       
]

def _vector3(x, as_type=None):
    if np.ndim(x) == 0: x = [x]*3
    elif np.ndim(x) != 1 or np.size(x) != 3:
        raise ValueError(f"cannot convert {x} to a 3-vector")
    return tuple(x) if as_type is None else tuple(as_type(xi) for xi in  x)

@dataclasses.dataclass(frozen=True, init=False, repr=False)
class Grid:
    pos     : npt.NDArray[np.float64]
    boxsize : tuple[float, float, float]
    origin  : tuple[float, float, float]
    cellsize: tuple[float, float, float]
    gridsize: tuple[int, int, int]      
    
    # Private data:
    _grid_data: npt.NDArray[np.int64]     
    _grid_info: npt.NDArray[np.int64]     

    def __init__(self, pos, boxsize, gridsize=None, cellsize=None, origin=(0., 0., 0.), nthreads=-1):
        
        origin  = _vector3(origin , as_type=float)
        boxsize = _vector3(boxsize, as_type=float)
        if not all(map(float.__gt__, boxsize, (0., 0., 0.))):
            raise ValueError("boxsize must be positive")

        if gridsize is not None and cellsize is None:
            
            gridsize = _vector3(gridsize, as_type=int)
            if not all(map(int.__gt__, gridsize, (0, 0, 0))):
                raise ValueError("gridsize must be positive")

            cellsize = tuple(x / y for x, y, in zip(boxsize, gridsize))

        elif gridsize is None and cellsize is not None:

            from math import floor

            cellsize = _vector3(cellsize, as_type=float)
            if not all(0. < x <= y for x, y in zip(cellsize, boxsize)):
                raise ValueError("cellsize must be positive and less than boxsize")

            gridsize = tuple(floor(x / y) for x, y in zip(boxsize, cellsize))

        else:
            raise ValueError("either gridsize or cellsize must be specified")

        for attr, value in [('boxsize' , boxsize ), ('origin'  , origin  ), 
                            ('gridsize', gridsize), ('cellsize', cellsize), ]:
            object.__setattr__(self, attr, value)

        boxinfo   = np.array((boxsize, origin, gridsize, cellsize), dtype=boxinfo_t)
        ncells    = gridsize[0]*gridsize[1]*gridsize[2]
        pos       = np.array(pos,          dtype='f8'   )
        grid_data = np.zeros(pos.shape[0], dtype='i8'   )
        grid_info = np.zeros(ncells      , dtype=cinfo_t)
        error     = np.array(-1, dtype='i4')
        nthreads  = os.cpu_count() if nthreads < 1 else nthreads
        pid       = os.getpid()
        lib.build_grid_hash(
            pid, 
            pos.shape[0], pos, 
            boxinfo, 
            ncells, grid_info, grid_data, 
            nthreads, 
            error, 
        )
        if error != 0: 
            raise RuntimeError(f"grid creation failed with error code {error}")

        object.__setattr__( self, 'boxsize'   , boxsize   )
        object.__setattr__( self, 'origin'    , origin    )
        object.__setattr__( self, 'gridsize'  , gridsize  )
        object.__setattr__( self, 'cellsize'  , cellsize  )
        object.__setattr__( self, 'pos'       , pos       )
        object.__setattr__( self, '_grid_data', grid_data )
        object.__setattr__( self, '_grid_info', grid_info )
        return
    
    def count_neighbours(self, other, r, autocorr=False, periodic=False, nthreads=-1, verbose=0):

        if not isinstance(other, Grid): 
            raise TypeError(f"object must be of type Grid")
        
        # if self is other: autocorr = True
        
        gridsize = self.gridsize
        ncells   = gridsize[0]*gridsize[1]*gridsize[2]
        boxinfo  = np.array((self.boxsize, self.origin, gridsize, self.cellsize), dtype=boxinfo_t)
        
        if np.ndim(r) == 0:
            rbins = np.array([0., r], dtype='f8')
        else: 
            rbins = np.array(r, dtype='f8')
        cnts = np.zeros(rbins.shape, dtype='i8')

        nthreads = os.cpu_count() if nthreads < 1 else nthreads
        pid      = os.getpid()
        error    = np.array(-1, dtype='i4')
        lib.count_pairs(
            pid, 
            int( autocorr ),
            int( periodic ),
            boxinfo,
            rbins.shape[0], rbins, cnts, 
            ncells, self.pos.shape[0] , self._grid_info,  self._grid_data,  self.pos, 
                   other.pos.shape[0], other._grid_info, other._grid_data, other.pos, 
            nthreads,
            verbose,
            error,
        )
        if error != 0: 
            raise RuntimeError(f"neighbour counting failed with error code {error}")

        return cnts[:-1]
# Module for halo model calculations.

__all__ = [ 'halo_massfunction', 'halo_bias', 'HaloModel' ]

import numpy as np, os.path
from ctypes import Structure, c_double as f8, c_int as i4, c_int64 as i8, POINTER, byref 
from numpy.ctypeslib import load_library, ndpointer
from typing import Literal, TypeVar
from numpy.typing import NDArray
from .powerspectrum import psargs_t

_T = TypeVar('_T')

lib = load_library('libhaloutils', os.path.dirname(__file__))

# Pointer to a float64 ndarray of given dimension 
def f8pointer(ndim): return ndpointer(dtype=np.float64, ndim=ndim, flags='C_CONTIGUOUS')

# C struct for holding arguments for halo mass-function or bias calculations
class hmfargs_t(Structure):
    _fields_ = [
        ( 'z'       , f8    ), ( 'lnm' , f8 ), ( 'H0'      , f8 ), ( 'Om0'   , f8 ),
        ( 'Delta_m' , f8    ), ( 's'   , f8 ), ( 'dlnsdlnm', f8 ), ( 'rho_m' , f8 ),
        ( 'param'   , f8*16 ),        
    ]

def halo_massfunction(
        lnm      : _T, 
        model    : str, 
        ps       : psargs_t, 
        Delta_m  : float = 200., 
        retval   : Literal['dn/dm', 'dn/dlnm', 'dn/dlog10m', 'fs'] = 'dn/dm' , 
        filt     : Literal['tophat', 'gauss']                      = 'tophat',
    ) -> _T:
    r"""
    Return the value of halo mass-function based on the specified model.

    Parameters
    ----------
    lnm : array_like
        Natural log of halo mass in Msun.

    model : str
        Model name. Must be one of the avilable names.

    ps : psargs_t
        Power spectrum model. Redshift and other cosmology parameters are used from 
        this.

    Delta_m : float, default=200
        Overdensity relative to mean background.

    retval : {'dn/dm', 'dn/dlnm', 'dn/dlog10m', 'fs'}, optional
        Specify the format of the mass-function to return.

    filt :{'tophat', 'gauss'}, optional
        Specify the smoothing filter.

    Returns
    -------
    hmf : array_like
        Halo mass-function is specified format. Mass-functions `'dn/dlnm'` and `'dn/dlog10m'` 
        have unit Mpc^-3, `'dn/dm'` has unit Mpc^-3 Msun^-1 and `'fs'` is unitless.

    s : array_like
        Matter variance corresponding to the halo mass.

    dlnsdlnm : array_like
        First log derivative of variance, w.r.to halo mass.
    
    """
    return_code = { 'dn/dm': 0, 'dn/dlnm': 1, 'dn/dlog10m': 2, 'fs': -1 }.get(retval, 0)
    filter_code = { 'tophat': 0, 'gauss': 1 }.get(filt, 0)

    setup_function_args      = [ POINTER(hmfargs_t), i4, f8pointer(2), i8, i4 ]
    setup_hmf_basic          = lib.setup_hmf
    setup_hmf_basic.argtypes = setup_function_args; setup_hmf_basic.restype = None

    available_models = {
        'press74'   : (      setup_hmf_basic  , lib.hmf_press74   ), 
        'sheth01'   : (      setup_hmf_basic  , lib.hmf_sheth01   ),
        'jenkins01' : (      setup_hmf_basic  , lib.hmf_jenkins01 ), 
        'reed03'    : (      setup_hmf_basic  , lib.hmf_reed03    ), 
        'tinker08'  : ( lib.setup_hmf_tinker08, lib.hmf_tinker08  ), 
        'courtin10' : (      setup_hmf_basic  , lib.hmf_courtin10 ),  
        'crocce10'  : (      setup_hmf_basic  , lib.hmf_crocce10  ),
    }
    if model not in available_models:
        all_models = ', '.join(available_models)
        raise ValueError(f"halo mass-function model {model!r} is not available: available models are {all_models}")

    (setup_function, model_function) = available_models.get(model)
    model_function.argtypes = [ POINTER(hmfargs_t), i4 ]; model_function.restype = f8
    setup_function.argtypes = setup_function_args       ; setup_function.restype = None

    args = hmfargs_t( z = ps.z, H0 = ps.H0, Om0 = ps.Om0, Delta_m = Delta_m )
    setup_function( byref(args), filter_code, ps._table, ps._table.shape[0], ps._table.shape[1] )

    def not_vectorized(lnm: float, args: hmfargs_t, filter_code: int, 
                       return_code: int, ps: psargs_t,              ) -> tuple[float, float, float]:
        args.lnm = lnm
        setup_function( byref(args), filter_code, ps._table, ps._table.shape[0], ps._table.shape[1] )
        res = model_function( byref(args), return_code )
        return res, args.s, args.dlnsdlnm
    
    vectorized = np.vectorize(not_vectorized, otypes=[np.float64]*3, excluded=range(1, 5))
    res, s, dlnsdlnm = vectorized(lnm, args, filter_code, return_code, ps)
    return res, s, dlnsdlnm

def halo_bias(
        lnm      : _T, 
        model    : str, 
        ps       : psargs_t, 
        Delta_m  : float                      = 200., 
        filt     : Literal['tophat', 'gauss'] = 'tophat',
    ) -> _T:
    r"""
    Return the value of halo bias based on the specified model.

    Parameters
    ----------
    lnm : array_like
        Natural log of halo mass in Msun.

    model : str
        Model name. Must be one of the avilable names.

    ps : psargs_t
        Power spectrum model. Redshift and other cosmology parameters are used from 
        this.

    Delta_m : float, default=200
        Overdensity relative to mean background.

    filt :{'tophat', 'gauss'}, optional
        Specify the smoothing filter for variance calculation.

    Returns
    -------
    hbf : array_like
        Halo bias values.
        
    s : array_like
        Matter variance corresponding to the halo mass.

    dlnsdlnm : array_like
        First log derivative of variance, w.r.to halo mass.
    
    """
    filter_code = { 'tophat': 0, 'gauss': 1 }.get(filt, 0)

    setup_function_args      = [ POINTER(hmfargs_t), i4, f8pointer(2), i8, i4 ]
    setup_hmf_basic          = lib.setup_hmf
    setup_hmf_basic.argtypes = setup_function_args; setup_hmf_basic.restype = None

    available_models = {
        'cole89'   : ( setup_hmf_basic  , lib.hbf_cole89   ), 
        'sheth01'  : ( setup_hmf_basic  , lib.hbf_sheth01  ),
        'tinker10' : ( setup_hmf_basic  , lib.hbf_tinker10 ), 
    }
    if model not in available_models:
        all_models = ', '.join(available_models)
        raise ValueError(f"halo bias model {model!r} is not available: available models are {all_models}")

    (setup_function, model_function) = available_models.get(model)
    model_function.argtypes = [ POINTER(hmfargs_t) ]; model_function.restype = f8
    setup_function.argtypes = setup_function_args   ; setup_function.restype = None

    args = hmfargs_t( z = ps.z, H0 = ps.H0, Om0 = ps.Om0, Delta_m = Delta_m )
    setup_function( byref(args), filter_code, ps._table, ps._table.shape[0], ps._table.shape[1] )

    def not_vectorized(lnm: float, args: hmfargs_t, filter_code: int, ps: psargs_t) -> float:
        args.lnm = lnm
        setup_function( byref(args), filter_code, ps._table, ps._table.shape[0], ps._table.shape[1] )
        res = model_function( byref(args) )
        return res, args.s, args.dlnsdlnm
    
    vectorized = np.vectorize(not_vectorized, otypes=[np.float64]*3, excluded=range(1, 4))
    res, s, dlnsdlnm = vectorized(lnm, args, filter_code, ps)
    return res, s, dlnsdlnm

############################################################################################################

class hmargs_t(Structure):
    r"""
    A halo model based on a 5-parameter HOD. 
    """
    lnm_min    : float # Minimum halo mass (in Msun) to have at least one central galaxy 
    sigma_m    : float # Width of the central galaxy transition range. (0 for a step function)
    lnm0       : float # Minimum halo mass (in Msun) to have satellite galaxies
    lnm1       : float # Scale factor for power law satellite count relation (Msun)
    alpha      : float # Index for the  power law satellite count relation
    scale_shmf : float # Scale parameter (subhalo mass-function)
    slope_shmf : float # Slope parameter (subhalo mass-function)
    z          : float # Redshift
    H0         : float # Hubble parameter
    Om0        : float # Total matter density parameter
    Delta_m    : float # Matter overdensity w.r.to mean background density
    dplus      : float # Growth factor at this redshift

    # Fields of the C structure
    _fields_ = [
        ( 'lnm_min', f8 ), ( 'sigma_m'   , f8 ), ( 'lnm0'      , f8 ), ( 'lnm1' , f8 ),
        ( 'alpha'  , f8 ), ( 'scale_shmf', f8 ), ( 'slope_shmf', f8 ), ( 'z'    , f8 ),
        ( 'H0'     , f8 ), ( 'Om0'       , f8 ), ( 'Delta_m'   , f8 ), ( 'dplus', f8 ),
    ]

    # Returns a pointer to the C struct
    def pointer(self): return byref(self)

    def __init__(
            self, 
            ps         : psargs_t,    lnm_min    : float,      sigma_m : float,       
            lnm0       : float,       lnm1       : float,      alpha   : float, 
            scale_shmf : float = 0.5, slope_shmf : float = 2., Delta_m : float = 200., 
        ) -> None:

        self.ps = ps
        dplus   = ps.dplus_z / ps.dplus_0
        super().__init__(
            lnm_min    = lnm_min   , sigma_m = sigma_m, lnm0       = lnm0      ,
            lnm1       = lnm1      , alpha   = alpha  , scale_shmf = scale_shmf,
            slope_shmf = slope_shmf, z       = ps.z   , H0         = ps.H0     ,
            Om0        = ps.Om0    , Delta_m = Delta_m, dplus      = dplus     ,
        )
        self._hmfspline = None # interpolation table for halo mass-function
        self._hbfspline = None # interpolation table for halo bias

    def lagrangian_r(self, lnm: _T) -> _T:
        r"""
        Calculate the Lagrangian radius for a halo.

        Parameters
        ----------
        lnm : array_like 
            Natural log of halo mass in Msun.

        Returns
        -------
        lnr : array_like
            Natural log of halo radius in Mpc

        """
        lib.lagrangian_r.argtypes = [ POINTER(hmargs_t), f8 ]
        lib.lagrangian_r.restype  = f8
        fn = np.vectorize(lib.lagrangian_r, excluded=[0])
        return fn( self.pointer(), lnm )

    def central_count(self, lnm: _T) -> _T:
        r"""
        Return the average count of central galaxies in a halo, given its mass. This will be a 
        sigmoid function with smoothness controlled by the ``sigmaM`` parameter. If it is 0, 
        then it will be a step function.

        Parameters
        ----------
        lnm : array_like 
            Natural log of halo mass in Msun.

        Returns
        -------
        nc : array_like
            
        """
        lib.central_count.argtypes = [ POINTER(hmargs_t), f8 ]
        lib.central_count.restype  = f8
        fn = np.vectorize(lib.central_count, excluded=[0])
        return fn( self.pointer(), lnm )
    
    def satellite_count(self, lnm: _T) -> _T:
        r"""
        Return the average count of satellite galaxies in a halo, given its mass. This will be 
        a power law specified by the parameters ``(M0, M1, alpha)``. 

        Parameters
        ----------
        lnm : array_like 
            Natural log of halo mass in Msun.

        Returns
        -------
        ns : array_like
            
        """
        lib.satellite_count.argtypes = [ POINTER(hmargs_t), f8 ]
        lib.satellite_count.restype  = f8
        fn = np.vectorize(lib.satellite_count, excluded=[0])
        return fn( self.pointer(), lnm )

    def subhalo_massfunction(self, x: _T, lnm: float) -> _T:
        r"""
        Calculate subhalo mass-function corresponding to a specific halo mass.

        Parameters
        ----------
        x : array_like
            Mass fraction of subhalos w.r.to parent halo.
        lnm : float 
            Natural log of halo mass in Msun.

        Returns
        -------
        shmf : array_like
            Value of the subhalo mass-function, without normalization. This correspond to the 
            probability distribution of the sub-halo mass values. Multiply this with the correct 
            normalization factor to get the actual mass-function.

        """
        lib.subhalo_mass_function.argtypes = [ POINTER(hmargs_t), f8, f8 ]
        lib.subhalo_mass_function.restype  = f8
        fn = np.vectorize(lib.subhalo_mass_function, excluded=[0, 2])
        return fn( self.pointer(), x, lnm )

    def halo_concentration(self, lnm: _T, filt: Literal['tophat', 'gauss'] = 'tophat') -> _T:
        r"""
        Calculate concentration parameter for a halo.

        Parameters
        ----------
        lnm : array_like 
            Natural log of halo mass in Msun.

        filt :{'tophat', 'gauss'}, optional
            Specify the smoothing filter for variance calculation.

        Returns
        -------
        c : array_like
            
        """
        lib.halo_concentration.argtypes = [ POINTER(hmargs_t), f8 ]
        lib.halo_concentration.restype  = f8
        fn = np.vectorize(lib.halo_concentration, excluded=[0])
        s  = np.sqrt(self.ps.spectral_moment( self.lagrangian_r(lnm), filt = filt ))
        return fn( self.pointer(), s )
    
    def set_massfunction(self, data: NDArray[np.float64]) -> None:
        r"""
        Set the halo mass-function data for calculations. 

        Parameters
        ----------
        data : ndarray of shape=(N,2)
            First column is the natural log of mass in Msun and second column is the 
            natural log of mass-function dn/dlnm.

        """
        assert np.ndim(data) == 2 and np.size(data, 1) > 1
        spline        = np.empty(( np.size(data, 1), 3 ), dtype=np.float64)
        spline[:, :2] = data

        lib.generate_cspline.argtypes = [ i8, f8pointer(2) ]
        lib.generate_cspline.restype  = None
        lib.generate_cspline( spline.shape[0], spline )
        self._hmfspline = spline
        return
    
    def set_bias(self, data: NDArray[np.float64]) -> None:
        r"""
        Set the halo bias data for calculations. 

        Parameters
        ----------
        data : ndarray of shape=(N,2)
            First column is the natural log of mass in Msun and second column is the 
            natural log of halo bias.
            
        """
        assert np.ndim(data) == 2 and np.size(data, 1) > 1
        spline        = np.empty(( np.size(data, 1), 3 ), dtype=np.float64)
        spline[:, :2] = data
        
        lib.generate_cspline.argtypes = [ i8, f8pointer(2) ]
        lib.generate_cspline.restype  = None
        lib.generate_cspline( spline.shape[0], spline )
        self._hbfspline = spline
        return
    
    def massfunction(
            self, 
            lnm    : _T, 
            retval : Literal['dn/dm', 'dn/dlnm', 'dn/dlog10m'] = 'dn/dm',
        ) -> _T:
        r"""
        Calculate the halo mass-function.

        Parameters
        ----------
        lnm : array_like
            Natural log of halo mass in Msun.

        retval : {'dn/dm', 'dn/dlnm', 'dn/dlog10m'}
            Format of the returned mass-function.
        
        Returns
        -------
        hmf : array_like
            Mass-functions `'dn/dlnm'` and `'dn/dlog10m'` have unit Mpc^-3 and `'dn/dm'` 
            has unit Mpc^-3 Msun^-1.

        """
        if self._hmfspline is None:
            raise ValueError("mass-function data is not available")
        
        lib.interpolate.argtypes = [ f8, i8, f8pointer(2) ]
        lib.interpolate.restype  = f8
        interpolate = np.vectorize(lib.interpolate, otypes=[np.float64], excluded=[1, 2])

        res = np.exp( interpolate( lnm, self._hmfspline.shape[0], self._hmfspline ) ) # dn/dlnm
        if retval == 'dn/dlog10m':
            return res / np.log(10.)
        if retval == 'dn/dm':
            return res / np.exp(lnm)
        return res
    
    def bias(self, lnm: _T) -> _T:
        r"""
        Calculate the halo bias.

        Parameters
        ----------
        lnm : array_like
            Natural log of halo mass in Msun.
        
        Returns
        -------
        hbf : array_like

        """
        if self._hbfspline is None:
            raise ValueError("bias function data is not available")
        
        lib.interpolate.argtypes = [ f8, i8, f8pointer(2) ]
        lib.interpolate.restype  = f8
        interpolate = np.vectorize(lib.interpolate, otypes=[np.float64], excluded=[1, 2])

        res = np.exp( interpolate( lnm, self._hbfspline.shape[0], self._hbfspline ) ) # dn/dlnm
        return res
    
    # TODO: check
    def average_halo_density(
            self, 
            mrange  : tuple[float, float] = (0, np.inf),
            abstol  : float               = 1e-08,
            reltol  : float               = 1e-08,
            maxiter : int                 = 100,
        ) -> float:
        r"""
        Return the average halo number density at current redshift.

        Parameters
        ----------
        mrange : (float, float), default=(0, inf)
            Mass range over which average is calculated (unit: Msun).

        abstol : float, default=1e-08
            Absolute tolerance to check convergence of integral.

        reltol : float, default=1e-08
            Relative tolerance to check convergence of integral.

        maxiter : int, default=100
            Maximum number of iterations.

        Returns
        -------
        retval : float
            Halo number density in Mpc^-3.

        """
        if self._hmfspline is None:
            raise ValueError("mass-function data is not available")
        
        lnma = max(self.lnm_min, np.log(max(mrange[0], 1e-08)))
        lnmb = min(self._hmfspline[-1, 0], np.log(mrange[1]))
        
        lib.average_halo_density.argtypes = [ POINTER(hmargs_t), f8, f8, i8, f8pointer(2), f8, f8, i8 ]
        lib.average_halo_density.restype  = f8
        res = lib.average_halo_density(
            self.pointer(), lnma, lnmb, self._hmfspline.shape[0], self._hmfspline, 
            abstol, reltol, maxiter, 
        )
        return res
    
    # TODO: check
    def average_galaxy_density(
            self, 
            mrange  : tuple[float, float] = (0, np.inf),
            abstol  : float               = 1e-08,
            reltol  : float               = 1e-08,
            maxiter : int                 = 100,
        ) -> float:
        r"""
        Return the average galaxy number density at current redshift.

        Parameters
        ----------
        mrange : (float, float), default=(0, inf)
            Mass range over which average is calculated (unit: Msun).

        abstol : float, default=1e-08
            Absolute tolerance to check convergence of integral.

        reltol : float, default=1e-08
            Relative tolerance to check convergence of integral.

        maxiter : int, default=100
            Maximum number of iterations.

        Returns
        -------
        retval : float
            Galaxy number density in Mpc^-3.

        """
        if self._hmfspline is None:
            raise ValueError("mass-function data is not available")
        
        lnma = max(self.lnm_min, np.log(max(mrange[0], 1e-08)))
        lnmb = min(self._hmfspline[-1, 0], np.log(mrange[1]))
        
        lib.average_galaxy_density.argtypes = [ POINTER(hmargs_t), f8, f8, i8, f8pointer(2), f8, f8, i8 ]
        lib.average_galaxy_density.restype  = f8
        res = lib.average_galaxy_density(
            self.pointer(), lnma, lnmb, self._hmfspline.shape[0], self._hmfspline, 
            abstol, reltol, maxiter, 
        )
        return res

    # TODO: check
    def average_satellite_frac(
            self, 
            mrange  : tuple[float, float] = (0, np.inf),
            abstol  : float               = 1e-08,
            reltol  : float               = 1e-08,
            maxiter : int                 = 100,
        ) -> float:
        r"""
        Return the average satellite fraction at current redshift.

        Parameters
        ----------
        mrange : (float, float), default=(0, inf)
            Mass range over which average is calculated (unit: Msun).

        abstol : float, default=1e-08
            Absolute tolerance to check convergence of integral.

        reltol : float, default=1e-08
            Relative tolerance to check convergence of integral.

        maxiter : int, default=100
            Maximum number of iterations.

        Returns
        -------
        retval : float

        """
        if self._hmfspline is None:
            raise ValueError("mass-function data is not available")
        
        lnma = max(self.lnm_min, np.log(max(mrange[0], 1e-08)))
        lnmb = min(self._hmfspline[-1, 0], np.log(mrange[1]))
        
        lib.average_satellite_frac.argtypes = [ POINTER(hmargs_t), f8, f8, i8, f8pointer(2), f8, f8, i8 ]
        lib.average_satellite_frac.restype  = f8
        res = lib.average_satellite_frac(
            self.pointer(), lnma, lnmb, self._hmfspline.shape[0], self._hmfspline, 
            abstol, reltol, maxiter, 
        )
        return res
    
    # TODO: check
    def average_galaxy_bias(
            self, 
            mrange  : tuple[float, float] = (0, np.inf),
            abstol  : float               = 1e-08,
            reltol  : float               = 1e-08,
            maxiter : int                 = 100,
        ) -> float:
        r"""
        Return the average galaxy bias at current redshift.

        Parameters
        ----------
        mrange : (float, float), default=(0, inf)
            Mass range over which average is calculated (unit: Msun).

        abstol : float, default=1e-08
            Absolute tolerance to check convergence of integral.

        reltol : float, default=1e-08
            Relative tolerance to check convergence of integral.

        maxiter : int, default=100
            Maximum number of iterations.

        Returns
        -------
        retval : float

        """
        if self._hmfspline is None:
            raise ValueError("mass-function data is not available")
        if self._hbfspline is None:
            raise ValueError("bias data is not available")
        
        lnma = max(self.lnm_min, np.log(max(mrange[0], 1e-08)))
        lnmb = min(self._hmfspline[-1, 0], self._hbfspline[-1, 0], np.log(mrange[1]))
        
        lib.average_galaxy_bias.argtypes = [ POINTER(hmargs_t), f8, f8, i8, f8pointer(2), i8, f8pointer(2), f8, f8, i8 ]
        lib.average_galaxy_bias.restype  = f8
        res = lib.average_galaxy_bias(
            self.pointer(), lnma, lnmb, self._hmfspline.shape[0], self._hmfspline,   
            self._hbfspline.shape[0], self._hbfspline, abstol, reltol, maxiter, 
        )
        return res
    
    # C struct for holding the arguments for galaxy catalog generation
    class cgargs_t(Structure):
        _fields_ = [
            ( 'lnm'    , f8   ), ( 'pos'    , f8*3 ), ( 'lnr'  , f8 ), ( 's'     , f8 ),
            ( 'c'      , f8   ), ( 'n_cen'  , i8   ), ( 'n_sat', i8 ), ( 'rstate', i8 ),
            ( 'boxsize', f8*3 ), ( 'offset' , f8*3 ),
        ]
        def pointer(self): return byref(self)

    def generate_galaxies(
            self, 
            lnm     : float,
            pos     : tuple[float, float, float] = ( 0.,  0.,  0.),
            boxsize : tuple[float, float, float] = (-1., -1., -1.), 
            offset  : tuple[float, float, float] = ( 0.,  0.,  0.),
            count   : int = None, 
            rstate  : int = None,
        ) -> NDArray[np.float64]:
        r"""
        Generate a catalog of galaxies in a halo, given its mass in Msun amd position coordinates
        in Mpc.

        Parameters
        ----------
        lnm : float
            Natural logarithm of the mass of the halo in Msun. 

        pos : (float, float, float), optional
            Position coordinates of the halo in Mpc. 

        boxsize : (float, float, float), optional
            Size of the bounding box for all halos (in a simulation).

        offset : (float, float, float), optional
            Coordinates of the lower left corner of the simulation box.

        count : int, optional
            If specified, generate that number of galaxies.

        rstate : int, optional
            Seed for the random number generator.

        Returns
        -------
        cat : ndarray of shape=(N, 4)
            Catalog of galaxies in the halo. First 3 columns are the position coordinates in 
            Mpc and the last column is the mass in Msun. This will have size `N=1 + n_sat`.  
            First item is the central galaxy and the remaining are satellites.
        
        """
        import time

        rstate   = rstate or int( time.time() )
        cgargs_t = self.cgargs_t
        sigma    = np.sqrt(self.ps.spectral_moment( self.lagrangian_r(lnm) ))
        args     = self.cgargs_t(
            lnm     = lnm    , 
            pos     = pos    ,
            s       = sigma  ,
            rstate  = rstate ,
            boxsize = boxsize,
            offset  = offset ,
        )

        lib.setup_catalog_generation.argtypes = [ POINTER(hmargs_t), POINTER(cgargs_t) ]
        lib.setup_catalog_generation.restype  = None
        
        lib.generate_galaxies.argtypes = [ POINTER(hmargs_t), POINTER(cgargs_t), i8, f8pointer(2) ]
        lib.generate_galaxies.restype  = None
        
        remaining = count or 1
        gdata     = [] 
        start     = 0
        while remaining > 0:
            lib.setup_catalog_generation( self.pointer(), args.pointer() )
            generated  = args.n_cen + args.n_sat
            _gdata_    = np.empty((generated, 4), dtype=np.float64)
            remaining -= (generated - args.n_cen*(1 - start))
            lib.generate_galaxies( self.pointer(), args.pointer(), generated, _gdata_ )
            gdata.append(_gdata_[start:, :])
            start      = 1
        
        return np.vstack(gdata)

HaloModel = hmargs_t

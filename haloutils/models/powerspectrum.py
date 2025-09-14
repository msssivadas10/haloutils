# Module for linear matter power spectrum and related calculations.  

__all__ = [ 'Eisenstein98_zb', 'Eisenstein98_mnu', 'Eisenstein98_bao' ]

import numpy as np, os.path
from ctypes import Structure, c_double as f8, c_int as i4, c_int64 as i8, POINTER, byref 
from numpy.ctypeslib import load_library, ndpointer
from typing import Literal, TypeVar, Callable

_T = TypeVar('_T')

lib = load_library('libhaloutils', os.path.dirname(os.path.dirname(__file__)))

# Pointer to a float64 ndarray of given dimension 
def f8pointer(ndim): return ndpointer(dtype=np.float64, ndim=ndim, flags='C_CONTIGUOUS')

class psargs_t(Structure):
    r"""
    Base class for linear matter power spectrum models.
    """
    z      : float # Redshift 
    H0     : float # Hubble parameter (H0 = 100h km/s/Mpc)
    Om0    : float # Total matter density parameter
    Ob0    : float # Baryon matter density parameter 
    Onu0   : float # Massive neutrino density parameter
    Ode0   : float # Dark-energy density parameter
    w0     : float # Dark-energy equation of state constant part
    wa     : float # Dark-energy equation of state time-varying part
    Nnu    : float # Number of massive neutrinos
    ns     : float # Initial index of power spectrum
    sigma8 : float # Power spectrum normalization (variance at 8 Mpc/h)
    Tcmb0  : float # Temperature of background radiation (Tcmb0 = 2.7theta K )
    dplus_z: float # Linear growth at this redshift
    dplus_0: float # Linear growth at redshift z=0 (for normalizing)
    
    # Other parameters (fields of the structure)
    h: float; Omh2: float; Obh2: float; Onuh2: float; theta: float  

    _setup_ : Callable
    _model_ : Callable

    # Fields of the C structure
    _fields_ = [
            ( 'z'         , f8  ), ( 'h'      , f8   ), ( 'Omh2'   , f8 ), ( 'Obh2'  , f8 ),
            ( 'Onuh2'     , f8  ), ( 'Nnu'    , f8   ), ( 'ns'     , f8 ), ( 'sigma8', f8 ),
            ( 'theta'     , f8  ), ( 'dplus_z', f8   ), ( 'dplus_0', f8 ), ( 'norm'  , f8 ),
            ( 'include_nu', i4  ), ( 'z_eq'   , f8   ), ( 'z_d'    , f8 ), ( 's'     , f8 ),
            ( 'k_silk'    , f8  ), ( 'param'  , f8*5 ),
    ]
    
    # Arguments for linear growth factor calculations
    zfargs_t = np.dtype([
        ( 'Om0'   , 'f8' ), ( 'Ode0'  , 'f8' ), ( 'w0'     , 'f8' ), ( 'wa', 'f8' ), 
        ( 'abstol', 'f8' ), ( 'reltol', 'f8' ), ( 'maxiter', 'i8' ), 
    ])
    # Setting up library function for linear growth calculation
    lib.linear_growth.argtypes = [ f8, i4, ndpointer(zfargs_t, ndim=0, flags='C_CONTIGUOUS') ] # (z, nu, args)
    lib.linear_growth.restype  = f8
    _linear_growth_ = np.vectorize( lib.linear_growth, otypes = [np.float64], excluded=[1, 2] )   

    # Setting up library function for variance / spectral moment calculation
    lib.variance.argtypes = [ f8, i4, i4, i4, f8pointer(2), i8, i4 ]
    lib.variance.restype  = f8
    _variance_ = np.vectorize(lib.variance, otypes=[np.float64], excluded=range(1,7))

    # Setting up library function for correlation calculation
    lib.correlation.argtypes = [ f8, i4, f8pointer(2), i8, i4 ]
    lib.correlation.restype  = f8
    _correlation_ = np.vectorize(lib.correlation, otypes=[np.float64], excluded=range(1,5))

    # Returns a pointer to the C struct
    def pointer(self): return byref(self)
    
    def linear_growth(
            self, 
            z      : _T, 
            nu     : bool  = False, 
            abstol : float = 1e-08, 
            reltol : float = 1e-08, 
            maxiter: int   = 100, 
        ) -> _T:
        r"""
        Calculate the linear growth factor, or its logarithmic derivative, in the specified 
        cosmology. 

        Parameters
        ----------
        z : array_like
            Redshift - values must be greater than -1.

        nu : bool, default=False
            If true, calculate the linear growth rate (log derivative of growth factor).

        abstol : float, default=1e-08
            Absolute tolerance for checking convergence of the integral.

        reltol : float, default=1e-08
            Relative tolerance for checking convergence of the integral.

        maxiter : int, default=100
            Maximum number of iterations.

        Returns
        -------
        retval : array_like
            Linear growth factor or its first log-derivative.

        """
        args = np.array(
            (self.Om0, self.Ode0, self.w0, self.wa, abstol, reltol, maxiter), 
            dtype=self.zfargs_t,
        )
        return self._linear_growth_(z, int(nu), args)
    
    def setup_variance_calculations(
            self, 
            krange: tuple[float, float] = (1e-04, 1e+04), 
            npts  : int                 = 1001,
        ) -> None:
        r"""
        Setup the power spectrum table for calculating the spectral moments.

        Parameters
        ----------
        krange : (float, float), default=(1e-04, 1e+04)
            Limits of k integration. Values are in Mpc^-1 unit.

        npts : int, default=1001
            Size of the table

        """
        ka, kb      = krange
        table       = np.empty((npts, 2), dtype=np.float64)
        table[:,0]  = np.linspace( np.log(ka), np.log(kb), npts ) # log(k)
        table[:,1]  = self._model_( table[:,0], self.pointer() )  # log( p(k) )
        self._table = table
        return
    
    def spectral_moment(
            self, 
            lnr : _T,
            j   : int = 0,
            nu  : int = 0,
            filt: Literal['tophat', 'gauss'] = 'tophat'
        ) -> _T:
        r"""
        Calculate the j-th spectral moment. j=0 gives the matter variance.

        lnr : array_like
            Scale argument. Natural log of scale in Mpc.

        j : int, default=0
            Order of the moment.

        nu : int, default=0
            Order of the derivative. Only supports 0-th and 1-st derivative.

        filt : {'tophat', 'gauss'}
            Smoothing filter.

        Returns
        -------
        sj : array_like

        """
        filt_id = { 'tophat': 0, 'gauss': 1 }.get(filt, 0)
        res     = self._variance_( lnr, j, nu, filt_id, self._table, self._table.shape[0], 2 )
        return res
    
    def correlation(
            self, 
            lnr : _T,
            nu  : int = 0,
        ) -> _T:
        r"""
        Calculate the matter correlation function.

        lnr : array_like
            Scale argument. Natural log of scale in Mpc.

        nu : int, default=0
            If non-zero, calculate the average correlation (this integral converges faster 
            than the normal correlation function integral).

        Returns
        -------
        xi : array_like

        """
        res = self._correlation_( lnr, nu, self._table, self._table.shape[0], 2 )
        return res
    
    def normalize(
            self, 
            krange: tuple[float, float] = (1e-04, 1e+04), 
            npts  : int                 = 1001,
            filt  : Literal['tophat', 'gauss'] = 'tophat', 
        ) -> None:
        r"""
        Normalize the power spectrum using sigma8 parameter.

        Parameters
        ----------
        krange : (float, float), default=(1e-04, 1e+04)
            Limits of k integration. Values are in Mpc^-1 unit.

        npts : int, default=1001
            Size of the table

        filt : {'tophat', 'gauss'}
            Smoothing filter.

        """
        # Setting the model to z=0: if the model is already initialised to a different 
        # redshift, then the current values are saved while calculations are done. 
        # After that, the old values are restored, to avoid the need for recalculating
        # the old values again. 
        old_values   = [ (field, getattr(self, field)) for field, _ in self._fields_ ]
        self.z       = 0. 
        self.dplus_z = self.dplus_0 
        self.sigma8  = 1. # normalization is for sigma8=1, to make the value sigma8 independent 
        self.norm    = 1.
        self._setup_( self.pointer() )

        # Calculating the power spectrum table for z=0:
        ka, kb     = krange
        table      = np.empty((npts, 2), dtype=np.float64)
        table[:,0] = np.linspace( np.log(ka), np.log(kb), npts ) # log(k)
        table[:,1] = self._model_( table[:,0], self.pointer() )  # log( p(k) )

        # Restore old values:
        for field, value in old_values: setattr(self, field, value)

        # Calculating variance at z=0. 
        filt_id             = { 'tophat': 0, 'gauss': 1 }.get(filt, 0)
        calculated_variance = self._variance_( np.log(8./self.h), 0, 0, filt_id, table, npts, 2 ) 
        self.norm           = 1. / calculated_variance
        return
    
    def power(self, lnk: _T) -> _T: 
        r"""
        Calculate the value of the linear matter power spectrum at current redshift.

        Parameters
        ----------
        lnk : array_like
            Scale at which the spectrum is evaluated. Natural logarithm of the wavenumber in 1/Mpc.

        Returns
        -------
        ps : array_like
            Power spectrum in Mpc^3 unit.  

        """
        y = self._model_( lnk, self.pointer() )
        return np.exp(y)
    
    def transfer(self, lnk: _T) -> _T:
        r"""
        Calculate the value of the linear matter transfer function at current redshift.

        Parameters
        ----------
        lnk : array_like
            Scale at which the spectrum is evaluated. Natural logarithm of the wavenumber in 1/Mpc.

        Returns
        -------
        tf : array_like
            Transfer function values.  

        """
        # Transfer function is calculated from the power spectrum, using the relation
        # P(k, z) = T(k, z)^2 * P_i(k), where the initial power spectrum is given by 
        # P_i(k) = sigma8^2 * Norm * k^ns.
        shift = 2*np.log(self.sigma8) + np.log(self.norm) + self.ns*lnk 
        y     = self._model_( lnk, self.pointer() ) - shift
        return np.exp(y)

    def __init__(
            self,  
            z          : float,        H0    : float,          Om0 : float,       Ob0 : float, 
            Ode0       : float,        Onu0  : float =  0.,    Nnu : float =  3., ns  : float =  1., 
            sigma8     : float =  1.,  Tcmb0 : float =  2.725, w0  : float = -1., wa  : float =  0., 
            include_nu : int   =  0 , 
        ) -> None:

        self.Om0, self.Ob0, self.Onu0, self.Ode0 = Om0, Ob0, Onu0, Ode0
        self.w0, self.wa    = w0, wa
        self.H0, self.Tcmb0 = H0, Tcmb0

        h       = H0 / 100.
        h2      = h**2
        dplus_0 = self.linear_growth(0)
        dplus_z = self.linear_growth(z)
        super().__init__(
            z          = z         , h       = h      , Omh2    = Om0*h2 , Obh2   = Ob0*h2, 
            Onuh2      = Onu0*h2   , Nnu     = Nnu    , ns      = ns     , sigma8 = sigma8, 
            theta      = Tcmb0/2.7 , dplus_z = dplus_z, dplus_0 = dplus_0, norm   = 1., 
            include_nu = include_nu, 
        )

        self.normalize()
        self._setup_( self.pointer() )
        self.setup_variance_calculations()
        return 
    
_PSBase = psargs_t
    
def _link_powerspectrum_model(model):

    available_models = {
        'eisenstein98_bao' : ( lib.init_eisenstein98_bao, lib.ps_eisenstein98_bao ),
        'eisenstein98_mnu' : ( lib.init_eisenstein98_mnu, lib.ps_eisenstein98_mnu ), 
        'eisenstein98_zb'  : ( lib.init_eisenstein98_zb , lib.ps_eisenstein98_zb  ),
    }

    setup_function, model_function = available_models.get(model)
    setup_function.argtypes = [     POINTER(psargs_t) ]; setup_function.restype = None
    model_function.argtypes = [ f8, POINTER(psargs_t) ]; model_function.restype = f8

    def _linker(cls):
        cls._setup_ = setup_function
        cls._model_ = np.vectorize(model_function, otypes=[np.float64], excluded=[1])
        return cls 
    
    return _linker

@_link_powerspectrum_model('eisenstein98_zb')
class Eisenstein98_zb(_PSBase):
    r"""
    A class implementing the power spectrum based on the linear matter transfer function by 
    Eisenetein & Hu (1998) with trace baryon content (no baryon oscillations).  

    """
    pass

@_link_powerspectrum_model('eisenstein98_mnu')
class Eisenstein98_mnu(_PSBase):
    r"""
    A class implementing the power spectrum based on the linear matter transfer function by 
    Eisenetein & Hu (1998) for non-zero massive neutrino content.  

    """
    pass

@_link_powerspectrum_model('eisenstein98_bao')
class Eisenstein98_bao(_PSBase):
    r"""
    A class implementing the power spectrum based on the linear matter transfer function by 
    Eisenetein & Hu (1998) with baryon oscillations.  

    """
    pass

#!/bin/bash

TASK_ID=12345 # $RANDOM

SIM_NAME="AbacusSummit_hugebase_c000_ph000" 
REDSHIFT=3
CATALOG_DIR="_data/abacus"
NPROCS=-1
RMIN=0.1
RMAX=100.0
NUM_RBINS=50
NUM_MBINS=20
JK_SUBDIVS=4
VENV=".venv"

source $VENV/bin/activate

# Calculating halo mass-function:
MASSFUNC_FILE="hmf.$SIM_NAME.z$(printf %04d $((100*REDSHIFT))).txt"
if [[ ! -e "$MASSFUNC_FILE" ]]; then
    # Mass bin edges:
    export BINS=$(python << EOF
import numpy as np
print( " ".join([ f"{x:8.4e}" for x in np.logspace(12, 15, 101) ]) )
EOF
)
    python haloutils/mass_function.py abacus-massfunction \
    --simname  $SIM_NAME      \
    --redshift $REDSHIFT      \
    --outfile  $MASSFUNC_FILE \
    --loc      $CATALOG_DIR   \
    --nprocs   $NPROCS
fi 

# Calculating mass bins: these mass bins are calculated in the range where 98%
# halos are present - this makes sure that the larger mass region with lesser 
# halos are avoided in correlation calculation... 
read -r -a M_EDGES <<< $(python << EOF
import numpy as np
from haloutils.misc.adaptive_binning import adaptive_bins
bins   = $NUM_MBINS
x, y   = np.loadtxt("$MASSFUNC_FILE", usecols=[0, 1], unpack=True)
cdf    = np.hstack([0., np.cumsum( np.diff(x) * (( y[1:] + y[:-1] ) / 2. - min(y)) )])
k      = cdf < 0.983*cdf[-1]
medges = np.exp( adaptive_bins(x[k], y[k], (max(x) - min(x)) / bins) )
print(' '.join([ f"{t:8.4e}" for t in medges ]))
EOF
) 
NUM_MBINS=$((${#M_EDGES[@]}-1))

# Distance bin edges:
export RBINS=$(python << EOF
import numpy as np 
redges = np.logspace( np.log10($RMIN), np.log10($RMAX), $NUM_RBINS + 1 )
print(' '.join([ f"{x:8.4e}" for x in redges ]))
EOF
)

# Combined correlation data:
CORRFUNC_FILE="xcf.$SIM_NAME.z$(printf %04d $((100*REDSHIFT))).asdf"
python << EOF
import numpy as np, asdf
redges   = np.fromstring("$RBINS", dtype="f8", sep=' ')
rcenters = np.sqrt( redges[1:]*redges[:-1] ) 
medges   = np.fromstring("${M_EDGES[@]}", dtype="f8", sep=' ')
mcenters = np.sqrt( medges[1:]*medges[:-1] ) 
xi_shape = [$NUM_MBINS, $NUM_MBINS, $NUM_RBINS]
with asdf.AsdfFile({
    "header": {}, 
    "data": {
        "powerSpectrum": [],
        "rcenters"     : rcenters, 
        "mcenters"     : mcenters,
        "xi"           : np.zeros(xi_shape, dtype="f8"), 
        "xiError"      : np.zeros(xi_shape, dtype="f8"),     
        "haloMassfunc" : np.loadtxt("$MASSFUNC_FILE", usecols=[0, 1])            
    }
}) as af:
    af.write_to( "$CORRFUNC_FILE", all_array_compression = 'zlib' )
EOF

CORR_WDIR="corr_$TASK_ID"
if [[ ! -d "$CORR_WDIR" ]]; then
    mkdir -p $CORR_WDIR
fi

# Correlation function calculations:
CORR_FILES=""
for ((I=0; I<$NUM_MBINS; I++)); do
    MA_I="${M_EDGES[I]}"
    MB_I="${M_EDGES[I+1]}"
    for ((J=$I; J<$NUM_MBINS; J++)); do
        MA_J="${M_EDGES[J]}"
        MB_J="${M_EDGES[J+1]}"
        CORR_FILE_IJ="$CORR_WDIR/$I.$J.asdf"
        python haloutils/correlation2.py abacus-corrfunc \
        --simname  $SIM_NAME      \
        --redshift $REDSHIFT      \
        --mrange1  $MA_I $MB_I    \
        --mrange2  $MA_J $MB_J    \
        --outfile  $CORR_FILE_IJ  \
        --workdir  $CORR_WDIR     \
        --loc      $CATALOG_DIR   \
        --nthreads $NPROCS        \
        --subdivs  $JK_SUBDIVS
        if [[ -e "$CORR_FILE_IJ" ]]; then
            # Calculating correlation function from counts and update final output:
            python << EOF
import numpy as np, asdf
from haloutils.correlation2 import correlation_from_count, pcresult
with asdf.open("$CORRFUNC_FILE") as afo, asdf.open("$CORR_FILE_IJ") as afi:
    afo["header"] = { 
        k: v 
            for k, v in afi["header"].items() 
            if k not in [ "jackknifeSamples", "mrange1", "mrange2" ]
    }
    xi      = correlation_from_count(pcresult( **afi["data"]["pairCounts_full"] ), "ls")
    xi_jack = correlation_from_count(pcresult( **afi["data"]["pairCounts_jack"] ), "ls")
    error   = np.sqrt( 
        ( len(xi_jack)-1 ) * np.mean(( xi_jack - np.mean(xi_jack, axis=0) )**2, axis=0) 
    )
    afo["data"]["xi"][i,j,:], afo["data"]["xiError"][i,j,:] = xi, error
    if i != j:
        afo["data"]["xi"][j,i,:], afo["data"]["xiError"][j,i,:] = xi, error
    afo["data"]["powerSpectrum"] = afi["data"]["powerSpectrum"]
    afo.write_to( "$CORRFUNC_FILE", all_array_compression = 'zlib' )
EOF
            CORR_FILES+="$CORR_FILE_IJ "
        fi
    done
done

# Create a zip file of pair counts:
CORRFUNC_ZIP="xcf.$SIM_NAME.z$(printf %04d $((100*REDSHIFT))).zip"
if [[ -n "$CORR_FILES" ]]; then
    zip $CORRFUNC_ZIP $CORR_FILES
fi 

# Clean-up: 
rm -rvf "$CORR_WDIR/*.bin"
# rm -rvf "$CORR_WDIR"

deactivate
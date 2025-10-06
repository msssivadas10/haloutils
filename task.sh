#!/bin/bash

SIM_NAME="AbacusSummit_hugebase_c000_ph000"
REDSHIFT=3
CATALOG_DIR="_data/abacus"
NPROCS=-1
RMIN=0.1
RMAX=50.0
NUM_RBINS=21
JK_SUBDIVS=4

TASK_ID=12345 # $RANDOM

source .venv/bin/activate

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

# Distance bin edges:
export RBINS=$(python << EOF
import numpy as np
print( " ".join([ f"{x:8.4e}" for x in np.logspace(np.log10($RMIN), np.log10($RMAX), $NUM_RBINS) ]) )
EOF
)

CORR_WDIR="corr_$TASK_ID"
if [[ ! -d "$CORR_WDIR" ]]; then
    mkdir -p $CORR_WDIR
fi

# Calculating mass bins:
MBINS_COUNT=5
read -r -a M_EDGES <<< $(python << EOF
import numpy as np
from numpy import loadtxt, hstack, cumsum, diff, exp
from haloutils.misc.adaptive_binning import adaptive_bins
bins = $MBINS_COUNT
x, y = loadtxt("$MASSFUNC_FILE", unpack=True)
cdf  = hstack([0., cumsum( diff(x) * (( y[1:] + y[:-1] ) / 2. - min(y)) )])
m    = cdf < 0.983*cdf[-1]
xb   = exp( adaptive_bins(x[m], y[m], (max(x) - min(x)) / bins) )
print(' '.join([f"{t:8.4e}" for t in xb]))
EOF
) 
MBINS_COUNT=$((${#M_EDGES[@]}-1))

# Correlation function calculations:
CORR_FILES="["
for ((I=0; I<$MBINS_COUNT; I++)); do
    MRANGE_I="${M_EDGES[I]} ${M_EDGES[I+1]}"
    for ((J=$I; J<$MBINS_COUNT; J++)); do
        MRANGE_J="${M_EDGES[J]} ${M_EDGES[J+1]}"
        CORRFUNC_FILE="$CORR_WDIR/$TASK_ID.$I.$J.asdf"
        python haloutils/correlation2.py abacus-corrfunc \
        --simname  $SIM_NAME      \
        --redshift $REDSHIFT      \
        --mrange1  $MRANGE_I      \
        --mrange2  $MRANGE_J      \
        --outfile  $CORRFUNC_FILE \
        --workdir  $CORR_WDIR     \
        --loc      $CATALOG_DIR   \
        --nthreads $NPROCS        \
        --subdivs  $JK_SUBDIVS
        if [[ -e "$CORRFUNC_FILE" ]]; then
            CORR_FILES+="(($I,$J),(${MRANGE_I// /,}),(${MRANGE_J// /,}),\"$CORRFUNC_FILE\"),"
        fi
    done
done
CORR_FILES+="]"

# Combining correlation functions:
CORRFUNC_FILE="xcf.$SIM_NAME.z$(printf %04d $((100*REDSHIFT))).asdf"
python << EOF
import numpy as np, asdf
from haloutils.correlation2 import correlation_from_count, pcresult

def geom_mean(x, y): 
    return np.sqrt(x*y)

def calc_error(samples):
    n    = len(samples)
    mean = np.mean( samples, axis=0 )
    return np.sqrt( ( n-1 ) * np.mean( (samples - mean)**2, axis=0 ) )

def combine_corr_data(files, outfile):
    if not files: return
    
    with asdf.open(files[0]) as af: 
        n_rbins = af["data"]["rbins"].shape[0]
        header = af["header"]
        for attr in [ "jackknifeSamples", "mrange1", "mrange2" ]:
            header.pop( attr, None ) 
        pktab    = np.array( af["data"]["powerSpectrum"] )
        rbins    = np.array( af["data"]["rbins"] )
        rcenters = np.sqrt( rbins[1:]*rbins[:-1] )
    im, jm    = max(ij for ij, *_ in files )
    value     = np.zeros((im+1, jm+1, rcenters.shape[0])) 
    error     = np.zeros_like(value)
    mcenters1 = np.zeros((im+1,))
    mcenters2 = np.zeros((jm+1,))
    for (i, j), file in files:
        with asdf.open(file) as af:
            header, data = af["header"], af["data"]
            mcenters1[i] = geom_mean( *header["mrange1"] )
            mcenters2[j] = geom_mean( *header["mrange2"] )
            value[i,j,:] = correlation_from_count(pcresult( **data["pairCounts_full"] ), "ls")
            error[i,j,:] = calc_error( correlation_from_count(pcresult( **data["pairCounts_jack"] ), "ls"))
            value[i,j,:] = value[j,i,:]
            error[i,j,:] = error[j,i,:]
   
    with asdf.AsdfFile({
        "header": header, 
        "data": {
            "powerSpectrum": pktab,
            "rcenters"     : rcenters, 
            "mcenters1"    : mcenters1,               
            "mcenters2"    : mcenters2,
            "xi"           : value, 
            "xiError"      : error,                 
        }
    }) as af:
        af.write_to( 
            outfile, 
            all_array_compression = 'zlib', 
        )
    return

combine_corr_data(
    files   = $CORR_FILES, 
    outfile = "$CORRFUNC_FILE", 
)
EOF

# Clean-up: 
# rm -rf "$CORR_WDIR"

deactivate
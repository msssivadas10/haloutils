
__version__ = "0.5a"

import numpy as np, numpy.typing as nt
import os, tempfile, shutil, multiprocessing, logging, asdf
from pathlib import Path
from collections import namedtuple
from typing import TypeVar, Literal, Callable
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

_T = TypeVar("_T")
pcres_t = namedtuple("pcres_t" , ["D1D2", "R1R2", "D1R2", "D2R1", "ND1", "ND2", "NR1", "NR2"])
def pair_count(
        D1       : _T,
        D2       : _T,
        R1       : _T,
        R2       : _T,
        rbins    : list[float],
        boxsize  : float, 
        periodic : bool  = False,
        subdivs  : int   =  3,
        nthreads : int   = -1,
    ) -> tuple[pcres_t, list[pcres_t]]:

    from math import ceil
    from operator import sub
    from itertools import product
    from Corrfunc.theory.DD import DD
    
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

    if nthreads < 1: nthreads = os.cpu_count()

    DD_counts, RR_counts = {}, {}
    DD_sizes , RR_sizes  = {}, {}  
    for i, j in box_pairs:
        if j < i: continue
        
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
            DD_sizes [(i, j)] = ( len(Di), len(Dj) )

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
            RR_sizes [(i, j)] = ( len(Ri), len(Rj) )

    DR_counts, RD_counts = {}, {}
    DR_sizes , RD_sizes  = {}, {}
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
            DR_sizes [(i, j)] = ( len(Di), len(Rj) )
        
        if D2_is_D1 and R2_is_R1: 
            RD_counts[(i, j)] = DR_counts[(i, j)]
            RD_sizes [(i, j)] = DR_sizes [(i, j)]
            continue

        if i in D2_box_info and j in R1_box_info:
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
            RD_sizes [(i, j)] = ( len(Di), len(Rj) )

    full_result = pcres_t(
        D1D2 = sum([ ( arr if i==j else 2*arr ) for (i, j), arr in DD_counts.items() ]), 
        R1R2 = sum([ ( arr if i==j else 2*arr ) for (i, j), arr in RR_counts.items() ]),
        D1R2 = sum([ arr for (i, j), arr in DR_counts.items() ]), 
        D2R1 = sum([ arr for (i, j), arr in RD_counts.items() ]), 
        ND1  = len(D1), 
        ND2  = len(D2),
        NR1  = len(R1), 
        NR2  = len(R2), 
    )

    jack_results = [
        pcres_t(
            D1D2 = (
                full_result.D1D2
                    - DD_counts[(k, k)]
                    - 2*sum([ arr for (i, j), arr in DD_counts.items() if k in (i, j) and i != j  ])
            ), 
            R1R2 = (
                full_result.R1R2
                    - RR_counts[(k, k)]
                    - 2*sum([ arr for (i, j), arr in RR_counts.items() if k in (i, j) and i != j  ])
            ),
            D1R2 = (
                full_result.D1R2 
                    - DR_counts[(k, k)]
                    - sum([ arr for (i, j), arr in DR_counts.items() if k in (i, j) and i != j  ])
            ), 
            D2R1 = (
                full_result.D2R1 
                    - RD_counts[(k, k)]
                    - sum([ arr for (i, j), arr in RD_counts.items() if k in (i, j) and i != j  ])
            ), 
            ND1  = full_result.ND1 + sub( *D1_box_info.get( k, (0, 0) ) ), 
            ND2  = full_result.ND2 + sub( *D2_box_info.get( k, (0, 0) ) ),
            NR1  = full_result.NR1 + sub( *R1_box_info.get( k, (0, 0) ) ), 
            NR2  = full_result.NR2 + sub( *R2_box_info.get( k, (0, 0) ) ), 
        )
        for k in range( subdivs**3 )
    ]
        
    return full_result, jack_results

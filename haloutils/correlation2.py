
__version__ = "0.5a"

import numpy as np, numpy.typing as nt
import os, tempfile, shutil, multiprocessing, logging, asdf
from pathlib import Path
from collections import namedtuple
from typing import TypeVar, Literal, Callable
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

_T = TypeVar("_T")

def _get_subbox_pairs(
        boxsize  : float, 
        r_max    : float, 
        subdivs  : int, 
        periodic : bool, 
    ) -> list[tuple[int, int]]:

    from math import ceil
    from itertools import product
    
    subboxsize = boxsize / subdivs
    near_boxes = ceil(r_max / subboxsize) # no. of boxes near to this box on each face
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

    return box_pairs

def _get_subbox_data(
        data    : _T, 
        boxsize : float, 
        subdivs : int,
        dir     : Path,
        prefix  : str, 
    ) -> tuple[list[_T], list[str]]:
    
    # Getting the integer sub-box index for each point
    subbox = np.floor((data + boxsize / 2) / (boxsize / subdivs)).astype("i8")  
    subbox = np.clip(subbox, 0, subdivs-1) # set points outside to sub-regions at edges 
    subbox = subbox[:,0]*subdivs**2 + subbox[:,1]*subdivs + subbox[:,2] # flat index

    subbox_data  = [None]*( subdivs**3 )
    memmap_files = []
    for i in range( subdivs**3 ):
        subbox_data_i = np.array( data[subbox == i] )
        n_points, *s  = np.shape(subbox_data_i, 0)
        if n_points > 4:
            # Large array: save data to a file and use memory-map
            fn = str( Path(dir).joinpath( f"{prefix}.{i}.bin" ) ) 
            subbox_data_i.astype("f8").tofile(fn)
            subbox_data_i = np.memmap(fn, dtype = "f8", mode = 'r', shape = (n_points, *s))
        subbox_data[i] = subbox_data_i
        memmap_files.append(fn)

    return subbox_data, memmap_files

def pair_count(
        D1       : _T,
        D2       : _T,
        R1       : _T,
        R2       : _T,
        rbins    : list[float],
        boxsize  : float, 
        periodic : bool  = False,
        subdivs  : int   = 3,
        nthreads : int   = -1,
        workdir  : Path  = None,
    ):

    from Corrfunc.theory.DD import DD

    D2_is_D1 = D2 is None or D2 is D1
    R2_is_R1 = R2 is None or R2 is R1

    subbox_D1, files = _get_subbox_data(D1, boxsize, subdivs, workdir, 'D1')
    if not D2_is_D1:
        subbox_D2, files_ = _get_subbox_data(D2, boxsize, subdivs, workdir, 'D2')
        files.extend(files_)
    else:
        subbox_D2 = subbox_D1
    
    subbox_R1, files_ = _get_subbox_data(R1, boxsize, subdivs, workdir, 'R1')
    files.extend(files_)
    if not R2_is_R1:
        subbox_R2, files_ = _get_subbox_data(R2, boxsize, subdivs, workdir, 'R2')
        files.extend(files_)
    else:
        subbox_R2 = subbox_R1

    subbox_pairs = _get_subbox_pairs(boxsize, max(rbins), subdivs, periodic)

    DD_counts = {}
    for i, j in subbox_pairs:
        if j < i: continue
        d1, d2 = subbox_D1[i], subbox_D2[j]
        counts = DD(
            X1 = d1[:,0], Y1 = d1[:,1], Z1 = d1[:,2], 
            X2 = d2[:,0], Y2 = d2[:,1], Z2 = d2[:,2], 
            autocorr = (i == j and D2_is_D1), 
            nthreads = nthreads, 
            binfile  = rbins, 
            periodic = periodic, 
            boxsize  = boxsize, 
            verbose  = False,
        )
        DD_counts[(i,j)] = counts["npairs"]

    RR_counts = {}
    for i, j in subbox_pairs:
        if j < i: continue
        r1, r2 = subbox_R1[i], subbox_R2[j]
        counts = DD(
            X1 = r1[:,0], Y1 = r1[:,1], Z1 = r1[:,2], 
            X2 = r2[:,0], Y2 = r2[:,1], Z2 = r2[:,2], 
            autocorr = (i == j and R2_is_R1), 
            nthreads = nthreads, 
            binfile  = rbins, 
            periodic = periodic, 
            boxsize  = boxsize, 
            verbose  = False,
        )
        RR_counts[(i,j)] = counts["npairs"]

    DR_counts = {}
    for i, j in subbox_pairs:
        d1, r2 = subbox_D1[i], subbox_R2[j]
        counts = DD(
            X1 = d1[:,0], Y1 = d1[:,1], Z1 = d1[:,2], 
            X2 = r2[:,0], Y2 = r2[:,1], Z2 = r2[:,2], 
            autocorr = 0, 
            nthreads = nthreads, 
            binfile  = rbins, 
            periodic = periodic, 
            boxsize  = boxsize, 
            verbose  = False,
        )
        DR_counts[(i,j)] = counts["npairs"]
        if not D2_is_D1 or not R2_is_R1:
            d2, r1 = subbox_D2[i], subbox_R1[j]
            counts = DD(
                X1 = d2[:,0], Y1 = d2[:,1], Z1 = d2[:,2], 
                X2 = r1[:,0], Y2 = r1[:,1], Z2 = r1[:,2], 
                autocorr = 0, 
                nthreads = nthreads, 
                binfile  = rbins, 
                periodic = periodic, 
                boxsize  = boxsize, 
                verbose  = False,
            )
            DR_counts[(i,j)] = ( counts["npairs"] + DR_counts[(i,j)] ) / 2

    for fn in files:
        try: os.remove(fn)
        except: pass

    DD_total = sum([ arr for arr in DD_counts.values() ])
    RR_total = sum([ arr for arr in RR_counts.values() ])
    DR_total = sum([ arr for arr in DR_counts.values() ])

    return


from Corrfunc.theory.DD import DD

def count1(data, boxsize, rbins, subdivs, periodic, nthreads):
    c1 = DD(
        X1 = data[:,0], Y1 = data[:,1], Z1 = data[:,2], 
        X2 = data[:,0], Y2 = data[:,1], Z2 = data[:,2], 
        autocorr = 0, 
        nthreads = nthreads, 
        binfile  = rbins, 
        periodic = periodic, 
        boxsize  = boxsize, 
        verbose  = False,
    )["npairs"]

    box = np.floor((data + boxsize / 2) / (boxsize / subdivs)).astype("i8")
    box = np.clip(box, 0, subdivs-1) # set points outside to sub-regions at edges 
    box = box[:,0]*subdivs**2 + box[:,1]*subdivs + box[:,2] # flattening index

    c2 = []
    for i in range(subdivs**3):
        data2 = np.array(data[ box != i ])
        c2_ = DD(
            X1 = data2[:,0], Y1 = data2[:,1], Z1 = data2[:,2], 
            X2 = data2[:,0], Y2 = data2[:,1], Z2 = data2[:,2], 
            autocorr = 0, 
            nthreads = nthreads, 
            binfile  = rbins, 
            periodic = periodic, 
            boxsize  = boxsize, 
            verbose  = False,
        )["npairs"]
        c2.append(c2_)
    c2 = np.array(c2)

    return c1, c2

def count2(data, boxsize, rbins, subdivs, periodic, nthreads):
    from math import ceil
    from itertools import product

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

    box = np.floor((data + boxsize / 2) / (boxsize / subdivs)).astype("i8")
    box = np.clip(box, 0, subdivs-1) # set points outside to sub-regions at edges 
    box = box[:,0]*subdivs**2 + box[:,1]*subdivs + box[:,2] # flattening index

    box_data = []
    for i in range(subdivs**3):
        data2 = np.array(data[ box == i ])
        box_data.append(data2)

    counts = {}
    for i, j in box_pairs:
        if j < i: continue
        data1, data2 = box_data[i], box_data[j]
        counts[(i,j)] = DD(
            X1 = data1[:,0], Y1 = data1[:,1], Z1 = data1[:,2], 
            X2 = data2[:,0], Y2 = data2[:,1], Z2 = data2[:,2], 
            autocorr = (i==j), 
            nthreads = nthreads, 
            binfile  = rbins, 
            periodic = periodic, 
            boxsize  = boxsize, 
            verbose  = False,
        )["npairs"]

    c1 = sum([ ( arr if i==j else 2*arr ) for (i, j), arr in counts.items() ])
    c2 = np.array([
        c1 - counts[(k, k)]
        - 2*sum([ arr for (i, j), arr in counts.items() if i != j and (i == k or j == k) ])
        for k in range(subdivs**3)
    ])

    return c1, c2

subdivs = 4
rbins = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
periodic = True
nthreads = 4
boxsize = 100
rng  = np.random.default_rng(123456)
data = rng.uniform(-boxsize / 2, boxsize / 2, (10000, 3))

c1, c2 = count1(data, boxsize, rbins, subdivs, periodic, nthreads)
d1, d2 = count2(data, boxsize, rbins, subdivs, periodic, nthreads)

# print(c1)
# print(d1)

print(c2[2])
print(d2[2])
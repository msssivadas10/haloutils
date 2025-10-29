import sys, os.path
sys.path.append( os.path.abspath( os.path.dirname(os.path.dirname(__file__)) ) )

import numpy as np, os
from Corrfunc.theory.DD import DD
from haloutils.correlation2 import pair_count_3d


generator = np.random.default_rng(1234567)

subdivs = 3
boxsize = 100.
particle_count = 1000
d1 = generator.uniform(-boxsize/2, boxsize/2, (  particle_count, 3))
r1 = generator.uniform(-boxsize/2, boxsize/2, (3*particle_count, 3))

rbins = np.array([0.1, 0.5, 1., 5., 10., 25])

pc_total, pc_jack = pair_count_3d(
    d1, None, r1, None, rbins, 
    boxsize=boxsize, periodic=True, subdivs=subdivs, nthreads=-1
)

npairs1 = pc_total.D1R2
npairs2 = DD(
    autocorr=0, nthreads=os.cpu_count(), binfile=rbins, 
    X1=d1[:,0], Y1=d1[:,1], Z1=d1[:,2], 
    X2=r1[:,0], Y2=r1[:,1], Z2=r1[:,2], 
    boxsize=boxsize, periodic=True,
)["npairs"]
# print(np.allclose(npairs1, npairs2))
assert np.allclose(npairs1, npairs2), "different counts"

def subbox_index(data, boxsize, subdivs):
    i = np.floor((data + boxsize / 2) / (boxsize / subdivs)).astype("i8")  
    i = np.clip(i, 0, subdivs-1) # set points outside to sub-regions at edges 
    i = i[:,0]*subdivs**2 + i[:,1]*subdivs + i[:,2] # flat index
    return i

d1_box = subbox_index(d1, boxsize, subdivs)
r1_box = subbox_index(r1, boxsize, subdivs)

k = 10 
for k in range(subdivs**3):
    d1_mask = (d1_box != k)
    r1_mask = (r1_box != k)
    npairs1 = pc_jack[k].D1R2
    npairs2 = DD(
        autocorr=0, nthreads=os.cpu_count(), binfile=rbins, 
        X1=d1[d1_mask,0], Y1=d1[d1_mask,1], Z1=d1[d1_mask,2], 
        X2=r1[r1_mask,0], Y2=r1[r1_mask,1], Z2=r1[r1_mask,2], 
        boxsize=boxsize, periodic=True,
    )["npairs"]
    # print(k, np.allclose(npairs1, npairs2))
    assert np.allclose(npairs1, npairs2), "different counts"
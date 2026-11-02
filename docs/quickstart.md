# Getting started

This page measures a tomographic shear two-point function on a HEALPix map
from scratch. Everything else in PixPoint follows the same pattern: build the
estimator for a footprint and a binning, find the pairs once, then measure as
many maps as you like.

## 1. What you need

- **Maps** in HEALPix RING ordering at one resolution `nside`. A shear map is
  an array `(nz, 2, npix)` holding $\gamma_1$ and $\gamma_2$ for each of `nz`
  tomographic bins; a scalar field (δ, κ) is `(nz, npix)`.
- **Weights**, `(nz, npix)`: how much each pixel counts in each bin. For a shear
  map built from a catalogue that is the summed shear weight (or the number of
  galaxies) that fell into the pixel; for a density contrast it is the
  coverage fraction or the random count; for a noiseless simulation it is 1
  everywhere inside the footprint. Pixels with zero weight contribute nothing.
- **A mask**, `(npix,)`, nonzero where the survey has data. Pixels outside it
  are never looked at, which is also what keeps the pair search small.

```python
import healpy as hp
import numpy as np

nside = 512
npix = hp.nside2npix(nside)
mask = hp.read_map("footprint.fits") > 0           # (npix,) bool

shear = np.load("shear.npy")                       # (nz, 2, npix): gamma_1, gamma_2
w = np.load("weights.npy")                         # (nz, npix)
```

## 2. Build the estimator and find the pairs

```python
from pixpoint import Global2PCF

g2 = Global2PCF(
    nside, mask=mask,
    nbins=20, theta_min=2.5, theta_max=250,        # 20 log bins from 2.5' to 250'
    device="auto",                                 # GPU if CuPy is installed, else CPU
)
g2.preprocess()
```

`preprocess()` finds every pair of unmasked pixels whose separation falls into
one of the bins and stores it. This is the expensive step, and it depends only
on the footprint and the binning, not on the maps: at `nside=512` on a
5000 deg² footprint it takes half a minute on a many-core machine and a few
minutes on a laptop, and it is done once.
Save the pairs to reuse them in later sessions:

```python
g2.save_pairs("pairs_512_20bins.h5")      # later: g2.load_pairs("pairs_512_20bins.h5")
```

## 3. Measure

```python
res = g2.gg(shear, w, flip_g1=True)
```

`flip_g1=True` multiplies $\gamma_1$ by −1 before measuring; whether you need
it depends on the sign convention your maps were made with (see
[Conventions](conventions.md)). The call returns a
{class}`~pixpoint.TwoPointResult`:

```python
res.theta            # (nbins,) geometric bin centres, arcmin
res.xi_p             # (ncomb, nbins) xi_plus  for every pair of tomographic bins
res.xi_m             # (ncomb, nbins) xi_minus
res.combinations     # ((0, 0), (0, 1), (0, 2), ..., (1, 1), ...) -- the row labels
```

With four tomographic bins there are ten combinations, in the order of
`res.combinations`: (0,0), (0,1), (0,2), (0,3), (1,1), ... Cross combinations
include both orientations of every pixel pair, so (0,1) is the usual symmetric
cross-correlation.

Plotting bin (1,1):

```python
import matplotlib.pyplot as plt

row = res.combinations.index((1, 1))
plt.loglog(res.theta, res.xi_p[row], label=r"$\xi_+$")
plt.loglog(res.theta, res.xi_m[row], label=r"$\xi_-$")
plt.xlabel(r"$\theta$ [arcmin]")
plt.legend()
```

## 4. Many maps

The pair search is paid; every further map is one call:

```python
results = [g2.gg(shear_k, w, flip_g1=True) for shear_k in simulation_maps]
xi_p = np.stack([r.xi_p for r in results])        # (nmaps, ncomb, nbins)
```

For long runs, stream the results to an HDF5 file instead of keeping them in
memory ([Files and streaming](io.md)), and on a GPU keep the maps on the
device between calls ([Running on GPUs](performance.md)).

## 5. The other fields

Same object, same pairs:

```python
wtheta = g2.kk(delta, w_cov).xi                     # clustering w(theta), delta (nz, npix)
gammat = g2.kg(delta, shear, w_lens, w_source).xi   # galaxy-galaxy lensing gamma_t
both = g2.three_by_two_pt(shear, delta, w_source, w_lens)   # {"gg": ..., "kk": ..., "kg": ...}
```

## Where next

- [Concepts](concepts.md): what exactly the estimator computes, how weights
  and masks enter, what the numbers mean.
- [Two-point functions](global2pcf.md): the full guide to `Global2PCF`.
- [Three-point functions](global3pcf.md), the [integrated 3PCF](integrated3pcf.md)
  and [one-point statistics](onepoint.md) for the other estimators.

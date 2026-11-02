# One-point statistics: `OnePoint`

{class}`~pixpoint.OnePoint` measures the *distribution* of an aperture field
over the footprint rather than a correlation: the probability density
function (PDF) and the moments of the aperture mass of a shear map, or of a
convergence or density map smoothed with a top hat or a compensated filter,
at several scales and per tomographic bin. It needs no pixel pairs.

## Setup

```python
import numpy as np
from pixpoint import OnePoint

op = OnePoint(
    nside, mask,
    scales=(20.0, 45.0, 90.0),            # aperture scales theta, arcmin
    min_coverage=0.8,                      # keep apertures with >= 80 % of their disc unmasked
    edges=np.linspace(-0.03, 0.03, 61),    # histogram edges (fixed up front, e.g. from simulations)
    device="auto",
)
```

For every scale $\theta$ the aperture centres are the pixels of a HEALPix grid
whose pixel is about $\theta/2$ (`centre_nside`), restricted to centres whose
aperture disc is at least `min_coverage` unmasked. The coverage is computed
from the mask alone, so the set of apertures does not depend on the weights or
on the window options, and `op.centres("g", i)` lists it.

## Measuring

```python
res = op.g(shear_maps, w, flip_g1=True)    # aperture mass M_ap and cross component M_x
res = op.k(kappa_maps, w)                  # kappa through a top hat of radius theta
```

`g` convolves the tangential shear about each centre with the compensated
filter $Q$ (Crittenden et al. 2002 by default, disc radius $5\theta$) and also
returns the cross component $M_\times$, the standard B-mode check. `k`
averages the scalar field over a disc of radius $\theta$; two alternatives are
set at construction:

```python
OnePoint(nside, mask, scales, k_aperture_filter="healpix")      # the HEALPix pixel of the centre as the top hat
OnePoint(nside, mask, scales, k_aperture_filter=U_crittenden)   # the aperture mass of kappa (compensated filter)
```

## The result

An {class}`~pixpoint.OnePointResult` holds **sums over the accepted apertures**
(unit weight each, the convention of the DES Y3 moments analysis), so results
of several GPUs or footprints add with `+`:

```python
res.n_apertures                 # (nscale,) accepted centres per scale
res.hist["M_ap"]                # (nz, nscale, nbins) counts on the half-open bins of edges
res.underflow["M_ap"], res.overflow["M_ap"]
res.pdf("M_ap")                 # histogram / (count * bin width)
res.moments("M_ap")             # (nz, nscale, 5): <x^k>, k = 1..5
res.central_moments("M_ap")     # (nz, nscale, 4): <(x - <x>)^k>, k = 2..5
res.cross_moments("M_ap")["ij"] # <x_i x_j> across tomographic bins, rows res.cross_index["ij"]
```

An aperture whose disc holds no weight in one of the tomographic bins is left
out of every bin's sums; `res.count` says how many apertures entered.
`values=True` returns the raw per-aperture values as well (`res.values[name]`,
one `(nz, n_apertures)` array per scale), useful to choose histogram edges
from a few simulations before fixing them.

## Streaming

```python
from pixpoint import OnePointWriter

with OnePointWriter("pdf.h5", op) as out:
    for shear in maps:
        out.submit(op.g(shear, w, flip_g1=True))
```

## Notes

- The per-aperture values are exactly the aperture statistics of
  {class}`~pixpoint.Integrated3PCF` with the centres as patches, so the two
  estimators share one definition of $M_{\rm ap}$.
- The apertures overlap heavily (centres $\theta/2$ apart, discs $5\theta$
  wide): the histogram counts are not independent, and the covariance of a
  PDF must come from simulations measured with the same centres.
- Mass mapping is not part of PixPoint: to measure a convergence PDF, produce
  κ upstream and pass it to `k`.
- Cost on an A100 at `nside=1024` for a DES-like footprint: 10 to 25 ms per
  scale and map, most of it in the aperture pass over about $2\times10^8$ disc
  pixels per scale.

## Reference

```{eval-rst}
.. autoclass:: pixpoint.OnePoint
   :members: g, k, prepare, centres, n_apertures, provenance
   :noindex:
```

# Three-point functions: `Global3PCF`

{class}`~pixpoint.Global3PCF` measures three-point correlation functions over
a whole footprint: the four shear components $\Gamma_0 \ldots \Gamma_3$, the
scalar three-point function $\zeta$ of a density contrast or convergence map,
and the mixed shear–scalar functions, for any combination of tomographic bins.
It uses the multipole method: every triangle is seen from its centre vertex,
the two legs run from the centre to the outer vertices with lengths in radial
bins $b_2$, $b_3$, and the dependence on the opening angle $\phi = \phi_3 - \phi_2$
is expanded in multipoles $e^{in\phi}$, which turns the triple sum into
sums over pairs about every centre.

## Building and preparing

```python
from pixpoint import Global3PCF

g3 = Global3PCF(nside, mask=mask, nbins=12, theta_min=15, theta_max=250,
                n_max=20, device="auto")
g3.preprocess()                       # pairs about every centre, once
```

| argument | meaning |
|---|---|
| `nbins`, `theta_min`, `theta_max` | log-spaced radial bins of the two legs, arcmin |
| `n_max` | the largest multipole kept; the opening-angle resolution of `binned()` |
| `device` | as everywhere; a list of GPU ids splits the centres across GPUs |

The pair search stores every ordered (centre, neighbour) pair with its full
position angle, which is more than the two-point search keeps: 12 bytes per
pair, about $1.3 \times 10^9$ pairs for DES Y3 at `nside=512` and 250′. The
three-point estimator runs at full resolution only.

## Measuring

```python
res = g3.ggg(shear_maps, w, flip_g1=True)                 # Gamma_0 .. Gamma_3
res = g3.kkk(delta_maps, w)                               # zeta of a scalar field
res = g3.kgg(delta_maps, shear_maps, w_lens, w_source)    # <k g g> and <k g* g>
res = g3.kkg(delta_maps, shear_maps, w_lens, w_source)    # <k k g>
```

and, for any combination of spin-0 and spin-2 fields and conjugations,

```python
from pixpoint import Component

res = g3.measure(
    {"k": (delta_maps, w_lens, 0), "g": (shear_maps, w_source, 2)},   # name: (maps, weights, spin)
    [Component("gkk*", fields=("g", "k", "k"), conj=(True, False, False))],
)
```

A {class}`~pixpoint.results.Component` says which field sits at the centre, on
leg 2 and on leg 3, and which of the three is complex-conjugated.

## The result

```python
res.num["gamma0"].shape     # (2 n_max + 1, nz_centre, nz_leg2, nz_leg3, nbins, nbins)
res.n                       # the multipole index of axis 0: -n_max .. n_max
res.norm["gamma0"]          # the same sums over the weights (the normalisation)
```

The multipoles are exact for every $n$; nothing is approximated until you ask
for the function in opening-angle bins:

```python
import numpy as np
gamma = res.binned(np.linspace(-np.pi, np.pi, 17))   # label -> (nz, nz, nz, nbins, nbins, 16)
```

which truncates the Fourier series at `n_max`. Results of several chunks,
footprints or GPUs add with `+` before `binned()`.

## Projection

Spin-2 values are reported in the **x-projection**: at every vertex the shear
is projected along the leg that connects the vertex to the centre, the
centre's shear along leg 2. Compare a theory prediction by converting it to
this projection per triangle rather than rotating the data; the full
definition and the conversion to the centroid projection are on the
[conventions](conventions.md) page and in the [estimator note](estimator_note.md).

## Cost

On an A100 the shear 3PCF of DES Y3 at `nside=512` (12 radial bins,
`n_max=20`, four tomographic bins, all 64 triplets, all four components)
takes about 1.6 s per map after the pair search; memory is the stored
ordered pairs, which is what limits the resolution (at `nside=1024` the pairs
of the same binning no longer fit a single GPU). On a CPU expect minutes per
map.

## Reference

```{eval-rst}
.. autoclass:: pixpoint.Global3PCF
   :members: preprocess, save_pairs, load_pairs, ggg, kkk, kgg, kkg, measure
   :noindex:

.. autoclass:: pixpoint.results.Component
   :noindex:
```

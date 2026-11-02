# Concepts

This page explains what PixPoint computes and how your inputs enter, without
any GPU vocabulary. If you know what a two-point correlation function is, you
know enough.

## Maps, not catalogues

A catalogue code like TreeCorr correlates objects: every galaxy with every
other galaxy within some separation. PixPoint correlates **pixels**: every
unmasked HEALPix pixel of your map with every other one whose separation falls
into one of your angular bins. The pixel's value is the field (the mean shear
of the galaxies in it, the density contrast, the convergence), and the pixel's
weight says how much it counts.

That is exactly what you get from a catalogue if you first bin the galaxies
onto a HEALPix grid: a pixel holding $N$ galaxies with shears $\gamma_a$ and
weights $w_a$ becomes one object with value $\sum_a w_a \gamma_a / \sum_a w_a$
and weight $\sum_a w_a$. The two-point function of those pixels is the
two-point function of the galaxies, smoothed on the pixel scale. For angular
separations larger than a few pixels the smoothing is negligible; a
pixel-pair estimator is the standard choice for simulated maps, mass maps and
any analysis whose forward model produces maps.

Because the pairs depend only on **which pixels are unmasked** and **which
bins you want**, they can be found once and reused for every map that lives
on the same footprint. Finding them is the slow step (seconds to minutes);
measuring a map afterwards is one pass over the stored pairs (milliseconds on
a GPU, seconds on a CPU).

## The estimators

All two-point estimators are ratios of weighted sums over the pixel pairs
$(i, j)$ whose separation $\theta_{ij}$ falls in bin $b$:

$$
\xi_\pm^{(b)} = \frac{\sum_{(i,j)\in b} w_i w_j\, \tilde\gamma_i \tilde\gamma_j^{(*)}}{\sum_{(i,j)\in b} w_i w_j},
\qquad
w(\theta)^{(b)} = \frac{\sum_{(i,j)\in b} w_i w_j\, \delta_i \delta_j}{\sum_{(i,j)\in b} w_i w_j},
\qquad
\gamma_{\rm t}^{(b)} = \frac{\sum_{(i,j)\in b} w_i w_j\, \delta_i\, \gamma_{{\rm t},j}(i)}{\sum_{(i,j)\in b} w_i w_j}.
$$

Here $\tilde\gamma$ is the shear rotated into the frame of the pair ($\xi_+$
uses the complex conjugate, $\xi_-$ does not), and $\gamma_{{\rm t},j}(i)$ is
the tangential component of the shear at pixel $j$ about pixel $i$. These are
the standard estimators (TreeCorr's `GG`, `KK` with $k=\delta$, and `KG`),
evaluated on pixels. The formal definitions, including the aperture and
three-point statistics, are in the [estimator note](estimator_note.md).

PixPoint returns the **sums** as well as the ratio: a
{class}`~pixpoint.TwoPointResult` holds `num["xi_p"]` ($\sum w w \tilde\gamma\tilde\gamma^*$)
and `weight` ($\sum w w$) per bin, and `xi_p` is their ratio. Sums add, so
results of two footprints, two GPUs or two chunks of a simulation can be
combined with `+` before the ratio is taken.

## Weights and masks

- **Weights** are per pixel and per tomographic bin, shape `(nz, npix)`. Use
  the quantity that makes the pixel a fair representative of what is in it:
  the summed shear weight or galaxy count for a shear map, the coverage
  fraction (or random count) for a density contrast map, 1 inside the
  footprint for a noiseless simulation. A zero weight removes the pixel from
  every pair it would have been in.
- **The mask** is one array `(npix,)` for the whole footprint, nonzero where
  there is data. Masked pixels are never part of a pair. The mask also
  defines the *row space* (below).
- The pair search uses the mask, not the weights, so one pair set serves
  maps with different weights, and a weight that varies between maps (shape
  noise realisations, different galaxy samples) needs no new pair search.

With coverage-fraction weights the `kk` estimator on δ is exactly the
Landy–Szalay estimator with randoms placed at the pixel centres.

## Tomography

Every map array carries a leading tomographic axis. `gg` on `nz` bins returns
all $n_z(n_z+1)/2$ combinations, `kg` returns every (lens bin, source bin)
pair. The results tell you which row is which:

```python
res.combinations        # ((0, 0), (0, 1), ..., (nz-1, nz-1)) for gg and kk
res_kg.combinations     # ((lens, source), ...) for kg
```

A cross combination $(i, j)$ sums both orientations of every pixel pair (pixel
$a$ in bin $i$ with pixel $b$ in bin $j$, and vice versa) into one numerator
and one denominator, which is the usual symmetric estimator. `auto_only=True`
on `kk` keeps only the diagonal, and `combinations=[(0, 1), (2, 3)]` measures a
subset.

## Row space: only the unmasked pixels

Internally PixPoint keeps only the unmasked pixels, in ascending RING order;
this is the *row space*. Every method accepts maps either as full-sky arrays
`(nz, [2,] npix)`, which it cuts down itself, or already in row space
`(nz, [2,] n_active)`:

```python
rows = g2.to_row_space(full_sky_maps)     # (nz, 2, n_active), do this once per archive
g2.row_pix                                # the HEALPix pixel index of every row
g2.row_pix_hash                           # fingerprint of (nside, mask) to store with an archive
```

For a large simulation suite, store the maps in row space: a DES-like
footprint covers an eighth of the sky, so the arrays are eight times smaller
and nothing has to be gathered per map.

## Bins and angles

Angular bins are logarithmic between `theta_min` and `theta_max` in
arcminutes. A result's `theta` is the geometric centre of each bin in
arcminutes; `binedges` are in radians. A pair is binned by the great-circle
separation of the two pixel centres, computed in double precision.

## Precision

Three independent choices, all keyword arguments of every estimator:

| argument | what it sets | recommendation |
|---|---|---|
| `map_precision` | the maps and weights as stored for the measurement | `"float32"` on a GPU (half the memory and traffic), `"float64"` to reproduce a double-precision reference exactly |
| `accumulation_precision` | the sums over pairs | `"float64"` whenever maps are float32: products are formed in single precision, summed in double, so no cancellation error accumulates |
| `rotation_precision` | the stored pair geometry (rotation angles) | `"float32"` (default); the angles are quantised to 1e-5 rad or better either way |

The pair *search* (which pair goes into which bin) always runs in double
precision unless asked otherwise, because single precision mis-bins a visible
fraction of the pairs near bin edges.

## Full resolution and the treecode

By default every bin is measured on the pixel pairs of the map as given. For
wide bins at coarse pixel scales the number of pairs grows as $\theta^2$, and a
*treecode* option (`resolution_factor`) measures the large separations on
cells of merged pixels instead. That is a controlled approximation with a
documented price; it is off by default and described in [its own page](treecode.md).

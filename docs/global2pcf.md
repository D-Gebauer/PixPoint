# Two-point functions: `Global2PCF`

{class}`~pixpoint.Global2PCF` measures the two-point correlation functions of
a whole footprint: the shear correlations $\xi_\pm(\theta)$, the clustering
correlation $w(\theta)$ of a density contrast (or any scalar field) and the
galaxy-galaxy lensing signal $\gamma_{\rm t}(\theta)$, tomographically, as
ratios of weighted sums over every pair of unmasked pixels. It is the
estimator most people want first.

## Building the estimator

```python
from pixpoint import Global2PCF

g2 = Global2PCF(
    nside,                        # HEALPix resolution of your maps
    mask=mask,                    # (npix,) nonzero = observed; None for the full sky
    nbins=20,                     # log-spaced angular bins ...
    theta_min=2.5, theta_max=250, # ... between these separations, in arcmin
    device="auto",                # "auto": GPU if CuPy is installed, else CPU
    map_precision="float32",      # see "Precision" below
    accumulation_precision="float64",
)
```

Only the first four arguments matter for *what* is measured. The rest control
how fast and where it runs and are explained on the
[performance](performance.md) and [treecode](treecode.md) pages; their defaults
measure the exact estimator on the pixel pairs as they are.

## Finding the pairs

```python
g2.preprocess()
```

finds every pair of unmasked pixels with a separation inside
`[theta_min, theta_max)` and sorts it into its bin. The cost depends on the
footprint, the resolution and the largest separation: for DES Y3 at
`nside=512` and bins up to 250′ there are $6.7 \times 10^8$ pairs, found in
about half a minute on a many-core machine (the search runs on all CPU cores)
and a few minutes on a laptop. The pairs are independent of the maps and of
the weights, so save them:

```python
g2.save_pairs("desy3_512_20bins.h5")
```

```python
g2 = Global2PCF(nside, mask=mask, nbins=20, theta_min=2.5, theta_max=250)
g2.load_pairs("desy3_512_20bins.h5")      # checks that the file matches this footprint and binning
```

A pair file is an HDF5 file; it stores enough about its footprint and binning
that loading it into an estimator built differently raises an error instead of
silently measuring something else. Memory: 24 bytes per pair in RAM and in the
file, 16 GB for the example above (see [performance](performance.md) for the
knobs that shrink it, down to 5.4 GB with `pack_host_pairs=True`).

## Measuring

### Shear: ξ±

```python
res = g2.gg(shear_maps, w, flip_g1=True)
```

| argument | shape | meaning |
|---|---|---|
| `shear_maps` | `(nz, 2, npix)` or `(nz, 2, n_active)` | $\gamma_1$ and $\gamma_2$ per tomographic bin |
| `w` | `(nz, npix)` or `(nz, n_active)` | weight of each pixel in each bin |
| `flip_g1`, `flip_g2` | bool | multiply a component by −1 first ([conventions](conventions.md)) |

The result:

```python
res.theta                 # (nbins,) bin centres in arcmin
res.xi_p, res.xi_m        # (ncomb, nbins)
res.combinations          # ((0,0), (0,1), ..., (nz-1, nz-1)): the rows
res.num["xi_p"], res.weight   # the sums behind the ratio, same shapes
res.to_numpy()            # a copy with host arrays (results on a GPU are device arrays)
```

### Clustering: w(θ)

```python
res = g2.kk(delta_maps, w_cov)          # delta_maps (nz, npix), w_cov (nz, npix)
res.xi                                  # (ncomb, nbins) = w(theta)
res = g2.kk(delta_maps, w_cov, auto_only=True)    # the nz auto-correlations only
```

`kk` is a *field* estimator: it correlates the values of the scalar field,
weighted. With `delta` the density contrast and `w` the coverage fraction of
each pixel (the fraction of its area inside the survey, or the expected count
from a random catalogue), it equals the Landy–Szalay estimator with randoms
at the pixel centres. The same method on a convergence map gives $\xi_{\kappa\kappa}$.

### Galaxy-galaxy lensing: γ_t

```python
res = g2.kg(delta_maps, shear_maps, w_lens, w_source, flip_g1=True)
res.xi                                  # (n_lens * n_source, nbins)
res.combinations                        # ((lens_bin, source_bin), ...)
res = g2.kg(delta, shear, w_lens, w_source, combinations=[(0, 2), (0, 3)])   # a subset
```

Lenses and sources can have different numbers of tomographic bins. The
estimator is the weighted mean tangential shear of the source pixels about the
lens pixels, weighted by $w_{\rm lens} \delta_{\rm lens} \cdot w_{\rm source}$;
there is no subtraction of a random-point signal, because on a map the mean
shear about the random pixel positions is what the weights of δ already
remove. If you want the cross component $\gamma_\times$ as a systematics
check, pass the shear map with its components swapped and `flip_g2=True`
(see the [one-point page](onepoint.md) for the identity used).

### All three at once: 3×2pt

```python
out = g2.three_by_two_pt(shear_maps, delta_maps, w_source, w_lens, flip_g1=True)
out["gg"].xi_p, out["kk"].xi, out["kg"].xi
```

One pass over the pairs for all three statistics, cheaper than three calls.

## Results are sums

A {class}`~pixpoint.TwoPointResult` holds the numerator and the weight sums
per bin; the estimates are formed on access. Two results of the same
estimator add:

```python
total = res_a + res_b          # e.g. two footprints, or two halves of a simulation
total.xi_p                     # the ratio of the summed sums
```

This is also how several GPUs combine their shares, and it means that you
should store the sums (`num`, `weight`) rather than the ratios if you intend
to combine measurements later.

## Comparing with TreeCorr

On a catalogue of the unmasked pixel centres with the pixel values and weights,
TreeCorr's `GGCorrelation`, `KKCorrelation` (with `k = δ`) and `KGCorrelation`
with `metric="Arc"`, `brute=True` and the same bin edges reproduce PixPoint's
ξ±, w(θ) and γ_t to rounding (both agree with an independent brute-force sum
to 1e-10 or better). With TreeCorr's default `bin_slop` the two differ by the
binning approximation TreeCorr makes; PixPoint at full resolution makes none.
A measured comparison, including timings, is in the repository's
`benchmarks/static_treecode/BENCHMARK_TREECORR.md`.

## Many maps

The point of the design: find pairs once, measure many maps.

```python
from pixpoint import ResultWriter, result_arrays

with ResultWriter("xi.h5", ResultWriter.collect_provenance(g2), group="gg") as out:
    for shear in maps:                                 # a simulation suite
        out.submit(result_arrays(g2.gg(shear, w, flip_g1=True)))
```

{class}`~pixpoint.ResultWriter` appends every result to an HDF5 file from a
background thread, with the estimator's provenance in the file, so a crash
loses at most the last few maps and nothing is held in memory ([Files and
streaming](io.md)). On a GPU, measuring from maps that already live on the
device avoids re-uploading them; the loaders on the [performance](performance.md)
page keep the GPU busy while the next map is read.

## Choosing the binning

- `theta_min`: a few pixel sizes. At `nside=512` a pixel is 6.9′ across, so
  separations below ~10′ are dominated by the pixelisation; measure them on a
  finer map instead of trusting them at 512.
- `theta_max`: the pair count grows as $\theta_{\max}^2$ and so do the pair
  search time and memory. Up to a few degrees is routine at `nside=512`; at
  `nside=2048` large separations are where the [treecode](treecode.md) earns
  its place.
- `nbins`: more bins cost nothing in the pair search and little in the
  measurement; the result arrays are tiny.

## Precision

The defaults (`map_precision="float64"`, `accumulation_precision="same"`) are
the safe choice and the one to use when you compare against a double-precision
reference. On a GPU, `map_precision="float32"` with
`accumulation_precision="float64"` halves memory and traffic: every product is
formed from single-precision map values but summed in double precision, so
the estimates differ from the double-precision ones at the 1e-7 level and show
no accumulation error even over a billion pairs.

## Several GPUs

```python
g2 = Global2PCF(nside, mask=mask, nbins=20, theta_min=2.5, theta_max=250, device=[0, 1])
g2.preprocess()                 # or load_pairs: each GPU takes its share
res = g2.gg(shear, w)           # host arrays in, host result out; equal to one GPU to 1e-13
```

Each GPU holds a part of the pairs and a copy of the maps, so a geometry that
does not fit one card is measured on two. See [Running on GPUs](performance.md).

## Reference

```{eval-rst}
.. autoclass:: pixpoint.Global2PCF
   :members: preprocess, save_pairs, load_pairs, gg, kk, kg, three_by_two_pt, to_row_space, row_pix, row_pix_hash, nbins, binedges, bincenters, ntotpairs, level_table
   :noindex:
```

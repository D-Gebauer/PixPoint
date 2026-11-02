# The treecode: measuring large separations on merged pixels

By default PixPoint measures every angular bin on the pixel pairs of the map
as given, and that is the right choice whenever it fits: it is exact, and on a
GPU it is fast. The number of pairs in a bin, however, grows as the square of
the separation, so at high resolution the widest bins dominate the memory and
the time while carrying the least information per pair: two pixels 200′ apart
do not need to be resolved at 1.7′ to measure their correlation.

The **static treecode** (`resolution_factor`) measures each bin on the coarsest
HEALPix level whose pixel is still at least $k$ times smaller than the bin's
inner edge: at $k = 4$ a bin starting at 60′ is measured on cells of at most
15′, i.e. on a map degraded to `nside=256` from `nside=1024`. A cell carries
the weight sum and the weighted mean of the pixels it merges, so the pair sums
of a bin are the exact sums over the cells. This cuts the pair count by orders
of magnitude at large separations and is what makes `nside=2048` or many
scales at once affordable.

```python
g2 = Global2PCF(nside, mask=mask, nbins=20, theta_min=2.5, theta_max=250,
                resolution_factor=4)          # True = the default k = 4; a number sets k
g2.level_table                                # which nside each bin is measured on
```

`cell_moments` (default `"quad"`) makes the cells carry, besides their mean,
the dipole and quadrupole of the shear about the cell centroid, so that the
rotation of every merged pair into its pair frame is corrected to second order
in the pixels' offsets. Without it, merging pixels suppresses $\xi_-$ by
$\langle \cos 4\delta\varphi \rangle$ (about 3 % at $k = 4$) and $\gamma_{\rm t}$ by
$\langle \cos 2\delta\varphi \rangle$; with it the residual is 20 to 50 times
smaller. $\xi_+$ and $w(\theta)$ have no phase and are not affected either
way. `"dipole"` is the first-order correction, `None` the plain cells.

## What it costs, and what it does not

The treecode is a **different, windowed estimator**, not a noisier version of
the exact one. Its effects are deterministic and identical for data and
simulations:

- Pairs near a bin edge are binned by the separation of their cell centroids,
  so some pairs land in the neighbouring bin: a binning window of about ±2 %
  in the amplitude of a bin at $k = 4$, alternating in sign where the level
  changes from one bin to the next (the data vector is not smooth across those
  seams).
- The remaining phase window on $\xi_-$ and $\gamma_{\rm t}$ after the moment
  correction is at the per-mille level at $k = 4$.
- On the integrated 3PCF with the DES Y3 mask at `nside=512`, $k \approx 3$
  shifts every element of the $\zeta_{a,\pm}$ data vector by less than
  0.3 σ of its patch-to-patch scatter, while $k = 1.5$ produces a coherent
  20 to 40 % suppression of $\zeta_{a,-}$ and is not safe.

Hence the rules:

1. **Use the same `resolution_factor` (and `cell_moments`) for data,
   simulations and covariance.** The window is part of the estimator;
   `level_table` records it and the file writers store it.
2. **Compare with theory through the window**, i.e. per bin with the pixel
   window of that bin's level, and do not interpolate a treecode data vector
   across a level seam.
3. The cells are built from the **binary mask**, not from the weights, so
   data and simulations with different weight distributions still share the
   same cells; they must share the same mask.
4. $k \gtrsim 3$ is safe for the i3PCF data vectors measured so far; $k = 4$
   is the default behind `resolution_factor=True`.

## Coarse apertures: `aperture_nside`

Independently of the pair statistics, the aperture statistics of
{class}`~pixpoint.Integrated3PCF` and {class}`~pixpoint.OnePoint` can be
evaluated on the map degraded to `aperture_nside`. That is a second,
independent window: it rescales the aperture mass almost uniformly across
scales (measured at the 0.05 σ level on the i3PCF when going from `nside=512`
to 128), where the treecode hits the small separations hardest. The same
moment-carrying cells correct the tangential phase and the filter's variation
across a coarse cell. The two knobs are not interchangeable and both belong in
the provenance of a data vector.

## When to use it

- You measure one map at full resolution: no need, unless it does not fit.
- You measure thousands of maps at `nside ≥ 1024` or with separations of
  several degrees: `resolution_factor=4` with the default `cell_moments`,
  stored with the data vector, checked once against a full-resolution
  measurement of the same maps.
- You compare against an exact code or need a smooth data vector: full
  resolution.

# PixPoint

**Correlation functions of HEALPix maps, fast.**

PixPoint measures two-point, three-point, integrated three-point and one-point
statistics of cosmological fields that live on HEALPix maps: cosmic shear, galaxy
clustering, galaxy-galaxy lensing, convergence. You give it maps and weights, it
gives you labelled results. It runs on an ordinary laptop CPU and, with the same
code, on NVIDIA GPUs, where a full-footprint tomographic shear 2PCF at DES Y3
resolution takes a few tens of milliseconds per map once the pairs are known.

```python
from pixpoint import Global2PCF

g2 = Global2PCF(nside=512, mask=mask, nbins=20, theta_min=2.5, theta_max=250)
g2.preprocess()                       # find the pixel pairs once

res = g2.gg(shear_maps, weights)      # shear_maps (nz, 2, npix), weights (nz, npix)
res.theta                             # bin centres in arcmin
res.xi_p, res.xi_m                    # (n_combinations, nbins)
res.combinations                      # which tomographic bins each row is
```

The pixel pairs are found once per footprint and binning and reused for every map
you measure afterwards, which is what makes PixPoint fast for simulation suites,
covariances and simulation-based inference: ten thousand maps cost ten thousand
measurements, not ten thousand pair searches.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting started
:link: quickstart
:link-type: doc
Install, measure a shear 2PCF on a map, understand the result.
:::

:::{grid-item-card} Two-point functions
:link: global2pcf
:link-type: doc
ξ±, w(θ) and γ_t over a whole footprint, tomography, 3×2pt, saving pairs.
:::

:::{grid-item-card} Three-point functions
:link: global3pcf
:link-type: doc
Shear and scalar 3PCFs by multipoles.
:::

:::{grid-item-card} Integrated 3PCF
:link: integrated3pcf
:link-type: doc
Aperture masses and 2PCFs in patches, the i3PCF.
:::

:::{grid-item-card} One-point statistics
:link: onepoint
:link-type: doc
PDFs and moments of aperture mass and smoothed fields.
:::

:::{grid-item-card} Running on GPUs
:link: performance
:link-type: doc
What a GPU buys you, memory, several GPUs, keeping it fed.
:::
::::

## What PixPoint measures

| estimator | measures | methods | result |
|---|---|---|---|
| {class}`~pixpoint.Global2PCF` | two-point functions of the whole footprint: ξ±, w(θ), γ_t | `gg`, `kk`, `kg`, `three_by_two_pt` | {class}`~pixpoint.TwoPointResult` |
| {class}`~pixpoint.Global3PCF` | three-point functions of the whole footprint by multipoles | `ggg`, `kkk`, `kgg`, `kkg`, `measure` | {class}`~pixpoint.ThreePointResult` |
| {class}`~pixpoint.Integrated3PCF` | aperture masses and 2PCFs in patches, the integrated 3PCF | `gg`, `kk`, `kg`, `three_by_two_pt`, `m_a`, `m_g` | {class}`~pixpoint.IntegratedResult` |
| {class}`~pixpoint.OnePoint` | PDFs and moments of aperture mass and smoothed scalar fields | `g`, `k` | {class}`~pixpoint.OnePointResult` |

Method names follow TreeCorr's field codes: `g` is a shear field (spin 2), `k`
a scalar field such as an overdensity δ or a convergence κ. `gg` is shear–shear,
`kk` scalar–scalar, `kg` scalar–shear (galaxy-galaxy lensing).

## How it compares with TreeCorr

TreeCorr correlates catalogues of objects; PixPoint correlates maps. If your
field is already pixelised (simulated shear maps, density contrast maps, mass
maps), PixPoint measures the same quantities TreeCorr would on a catalogue of the
pixel centres with the pixel values as weights, and agrees with it to rounding
on the exact estimators. It is built for the case where one footprint is
measured many times: the pair search is done once, every map after that is one
pass over the stored pairs, on a GPU if you have one. Where you have a catalogue
and measure it once, use TreeCorr.

```{toctree}
:hidden:
:caption: User guide

install
quickstart
concepts
global2pcf
global3pcf
integrated3pcf
onepoint
treecode
performance
io
conventions
```

```{toctree}
:hidden:
:caption: Reference

estimator_note
api/index
changelog
citing
contributing
```

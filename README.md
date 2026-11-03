# PixPoint

[![CI](https://github.com/D-Gebauer/PixPoint/actions/workflows/ci.yml/badge.svg)](https://github.com/D-Gebauer/PixPoint/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/D-Gebauer/PixPoint/graph/badge.svg?token=F4JC08UEJP)](https://codecov.io/github/D-Gebauer/PixPoint)
[![Documentation](https://readthedocs.org/projects/pixpoint/badge/?version=latest)](https://pixpoint.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Correlation functions of HEALPix maps, fast.**

<p align="center">
  <a href="https://pixpoint.readthedocs.io"><b>Documentation</b></a> ·
  <a href="https://pixpoint.readthedocs.io/en/latest/quickstart.html">Getting started</a> ·
  <a href="https://pixpoint.readthedocs.io/en/latest/install.html">Installation</a> ·
  <a href="https://pixpoint.readthedocs.io/en/latest/citing.html">Citing</a>
</p>

PixPoint measures two-point, three-point, integrated three-point and one-point
statistics of cosmological fields on HEALPix maps: cosmic shear, galaxy
clustering, galaxy-galaxy lensing, convergence. You give it maps and weights,
it gives you labelled results. The pixel pairs of a footprint are found once
and reused for every map, so a simulation suite costs one pair search and one
fast pass per map, on a CPU or, with the same code, on NVIDIA GPUs.

```python
from pixpoint import Global2PCF

g2 = Global2PCF(nside=512, mask=mask, nbins=20, theta_min=2.5, theta_max=250)
g2.preprocess()                          # find the pixel pairs once (save_pairs / load_pairs to reuse)

res = g2.gg(shear_maps, weights)         # shear_maps (nz, 2, npix), weights (nz, npix)
res.theta                                # bin centres, arcmin
res.xi_p, res.xi_m                       # (n_combinations, nbins)
res.combinations                         # which tomographic bins each row is

g2.kk(delta_maps, w_cov).xi              # clustering w(theta)
g2.kg(delta_maps, shear_maps, w_lens, w_source).xi   # galaxy-galaxy lensing gamma_t
```

## What it measures

| estimator | measures | methods | result |
|---|---|---|---|
| `Global2PCF` | two-point functions of the whole footprint: ξ±, w(θ), γ_t | `gg`, `kk`, `kg`, `three_by_two_pt` | `TwoPointResult` |
| `Global3PCF` | three-point functions of the whole footprint, by multipoles | `ggg`, `kkk`, `kgg`, `kkg`, `measure` | `ThreePointResult` |
| `Integrated3PCF` | aperture masses and 2PCFs in patches: the integrated 3PCF | `gg`, `kk`, `kg`, `three_by_two_pt`, `m_a`, `m_g` | `IntegratedResult` |
| `OnePoint` | PDFs and moments of aperture mass and smoothed scalar fields | `g`, `k` | `OnePointResult` |

Method names follow TreeCorr's field codes: `g` is a shear field, `k` a scalar
field such as a density contrast or convergence map. Every estimator is
tomographic, and every result carries the labels of its rows.

## Why PixPoint

- **Exact pixel-pair estimators.** The two-point functions are ratios of
  weighted sums over every pair of unmasked pixels, and agree with TreeCorr on
  the pixel centres to rounding.
- **Find pairs once, measure many maps.** A full-footprint tomographic shear
  2PCF of DES Y3 at nside 512 (6.7 × 10⁸ pairs) takes 33 ms per map on an A100
  after a one-off pair search; the same code runs on a laptop CPU.
- **Everything in one vocabulary.** Shear, clustering, galaxy-galaxy lensing
  and 3×2pt from one object; 3PCF multipoles for any mix of spin-0 and spin-2
  fields; the integrated 3PCF with its aperture filters; aperture-mass PDFs
  and moments.
- **Built for suites.** Streaming HDF5 writers with provenance and resume,
  loaders that hide the upload behind the measurement, multi-GPU splitting,
  and an optional treecode that cuts the pair count at large separations.
- **Checked.** Every GPU kernel has a NumPy twin and a brute-force reference
  test at machine precision; `mypy --strict` clean; ships `py.typed`.

## Installation

Python 3.12 or newer:

```bash
pip install git+https://github.com/D-Gebauer/PixPoint.git
```

For GPUs, install [CuPy](https://docs.cupy.dev/en/stable/install.html) for
your CUDA version in the same environment (`pip install cupy-cuda12x`, CUDA 12
or newer); PixPoint picks the GPU automatically. Details, including the
development install, in the
[installation guide](https://pixpoint.readthedocs.io/en/latest/install.html).

## Learn more

- [Concepts](https://pixpoint.readthedocs.io/en/latest/concepts.html): maps
  versus catalogues, weights and masks, what the estimators compute.
- [Two-point functions](https://pixpoint.readthedocs.io/en/latest/global2pcf.html),
  [three-point functions](https://pixpoint.readthedocs.io/en/latest/global3pcf.html),
  [integrated 3PCF](https://pixpoint.readthedocs.io/en/latest/integrated3pcf.html),
  [one-point statistics](https://pixpoint.readthedocs.io/en/latest/onepoint.html).
- [Running on GPUs](https://pixpoint.readthedocs.io/en/latest/performance.html)
  and [files and streaming](https://pixpoint.readthedocs.io/en/latest/io.html).
- [Conventions](https://pixpoint.readthedocs.io/en/latest/conventions.html)
  (shear signs, bin labels, the 3PCF projection) and the formal
  [estimator definitions](https://pixpoint.readthedocs.io/en/latest/estimator_note.html).

PixPoint was called CosmoFuse before version 8.0.0; `import CosmoFuse` is now
`import pixpoint`, nothing else in the API changed, and old pair files are
still read.

## Citing

Until PixPoint has a software paper of its own, please cite the paper it was
developed for, Gebauer et al. 2026, *SBi3PCF: simulation-based inference with
the integrated 3PCF*, JCAP 06 (2026) 036
([arXiv:2510.13805](https://arxiv.org/abs/2510.13805)), and name the PixPoint
version you used. Please also cite the papers behind the estimators you use.
The BibTeX entry and those papers are on the
[Citing](https://pixpoint.readthedocs.io/en/latest/citing.html) page.

## Contributing

Issues and pull requests are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md). PixPoint is released under the
[MIT licence](LICENSE).

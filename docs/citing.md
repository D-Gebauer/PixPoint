# Citing

If PixPoint contributes to a publication, please cite it and the papers behind
the estimators you used.

## The software

PixPoint does not have a software paper of its own yet. Until it does, please
cite the paper it was developed for, and name the PixPoint version you used:

- Gebauer, Halder, Seitz & Anbajagane 2026, *SBi3PCF: simulation-based
  inference with the integrated 3PCF*, JCAP 06 (2026) 036,
  [doi:10.1088/1475-7516/2026/06/036](https://doi.org/10.1088/1475-7516/2026/06/036)
  ([arXiv:2510.13805](https://arxiv.org/abs/2510.13805)).

```bibtex
@article{Gebauer2026,
  author        = {Gebauer, David and Halder, Anik and Seitz, Stella and Anbajagane, Dhayaa},
  title         = {{SBi3PCF: simulation-based inference with the integrated 3PCF}},
  journal       = {Journal of Cosmology and Astroparticle Physics},
  year          = {2026},
  volume        = {2026},
  number        = {06},
  pages         = {036},
  doi           = {10.1088/1475-7516/2026/06/036},
  eprint        = {2510.13805},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO}
}
```

Every release is archived on [Zenodo](https://doi.org/10.5281/zenodo.23915406),
and each version has a DOI of its own. To identify the exact version you used,
cite its DOI as well, from the version list of the Zenodo record (8.0.0:
[10.5281/zenodo.23915407](https://doi.org/10.5281/zenodo.23915407)). The DOI
[10.5281/zenodo.23915406](https://doi.org/10.5281/zenodo.23915406) stands for
all versions and resolves to the latest one.

```bibtex
@software{PixPoint,
  author    = {Gebauer, David},
  title     = {{PixPoint: correlation functions of HEALPix maps on GPUs and CPUs}},
  publisher = {Zenodo},
  version   = {8.0.0},
  doi       = {10.5281/zenodo.23915407},
  url       = {https://doi.org/10.5281/zenodo.23915407}
}
```

## The statistics

- **Integrated 3PCF**: Halder, Friedrich, Seitz & Varga 2021, *The integrated
  3-point correlation function of cosmic shear*, MNRAS 506, 2780
  ([arXiv:2102.10177](https://arxiv.org/abs/2102.10177)); Halder et al. 2023,
  *Beyond 3×2-point cosmology: the integrated shear and galaxy 3-point
  correlation functions* ([arXiv:2305.17132](https://arxiv.org/abs/2305.17132)),
  whose aperture-count definition (eq. 3.11) `M_g` follows.
- **Aperture filters**: Crittenden, Natarajan, Pen & Theuns 2002, ApJ 568, 20
  ([astro-ph/0012336](https://arxiv.org/abs/astro-ph/0012336)) for
  `Q_crittenden` / `U_crittenden`; Schneider, van Waerbeke, Jain & Kruse 1998,
  MNRAS 296, 873 ([astro-ph/9708143](https://arxiv.org/abs/astro-ph/9708143))
  for `Q_schneider` / `U_schneider`.
- **Multipole three-point estimator**: Chen & Szapudi 2005, ApJ 635, 743
  ([astro-ph/0508316](https://arxiv.org/abs/astro-ph/0508316)); Slepian &
  Eisenstein 2015, MNRAS, [doi:10.1093/mnras/stv2119](https://doi.org/10.1093/mnras/stv2119)
  ([arXiv:1506.02040](https://arxiv.org/abs/1506.02040)); Philcox et al. 2022,
  MNRAS, [doi:10.1093/mnras/stab3025](https://doi.org/10.1093/mnras/stab3025)
  ([arXiv:2105.08722](https://arxiv.org/abs/2105.08722)).
- **Spin-2 multipoles, the x-projection and the removal of the degenerate
  terms**: Porth, Heydenreich, Burger, Linke & Schneider 2024, A&A 689, A227
  ([arXiv:2309.08601](https://arxiv.org/abs/2309.08601)).
- **Natural components and the centroid projection of the shear 3PCF**:
  Schneider & Lombardi 2003, A&A 397, 809
  ([astro-ph/0207454](https://arxiv.org/abs/astro-ph/0207454)).
- **Treecode**: Barnes & Hut 1986, Nature 324, 446.

## Conventions and comparison code

- **TreeCorr** (field-code vocabulary, shear conventions, the reference the
  exact estimators are checked against): Jarvis, Bernstein & Jain 2004,
  MNRAS 352, 338 ([astro-ph/0307393](https://arxiv.org/abs/astro-ph/0307393)).

## Infrastructure

- **HEALPix**: Górski et al. 2005, ApJ 622, 759
  ([astro-ph/0409513](https://arxiv.org/abs/astro-ph/0409513)).
- **healpy**: Zonca et al. 2019, JOSS 4, 1298
  ([doi:10.21105/joss.01298](https://doi.org/10.21105/joss.01298)).
- **Numba** and **CuPy**: see their own citation guidance
  ([Numba](https://numba.pydata.org/numba-doc/latest/user/faq.html#how-do-i-reference-cite-acknowledge-numba-in-other-work),
  [CuPy](https://cupy.dev/)).

The repository's `CITATION.cff` holds the same citation in a machine-readable
form (GitHub's "Cite this repository" button).

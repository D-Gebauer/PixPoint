# Integrated three-point functions: `Integrated3PCF`

The integrated three-point correlation function (i3PCF; Halder et al. 2021,
2023) is the correlation, across many circular patches of the sky, between an
aperture statistic at the patch centre and a two-point function measured
inside the patch:

$$
\zeta_{a,\pm}(\theta) = \langle M_{\rm ap}\, \xi_\pm(\theta) \rangle_{\rm patches},
\qquad
\zeta_{g,g}(\theta) = \langle M_g\, w(\theta) \rangle_{\rm patches},
\qquad
\zeta_{g,{\rm t}}(\theta) = \langle M_g\, \gamma_{\rm t}(\theta) \rangle_{\rm patches}.
$$

It probes squeezed bispectrum configurations with two-point machinery.
{class}`~pixpoint.Integrated3PCF` measures the ingredients per patch (the
aperture mass $M_{\rm ap}$ or aperture count $M_g$ at the centre, and $\xi_\pm$,
$w$, $\gamma_{\rm t}$ from the pixel pairs inside the patch) and the helpers
form the eight i3PCF variants from them.

## Patches

Patches are discs of radius `patch_size` (arcminutes) around centres you
choose or let PixPoint choose from the mask:

```python
from pixpoint import Integrated3PCF

corr = Integrated3PCF.from_mask(
    nside, mask, nside_centers=32,        # candidate centres on a HEALPix grid at nside 32
    patch_size=90, theta_Q=90,            # patch radius and aperture filter scale, arcmin
    f_mask=0.2,                           # accept a patch if <= 20 % of its disc is masked
    nbins=10, theta_min=10, theta_max=170,
    device="auto",
)
```

or with explicit centres `Integrated3PCF(nside, phi_center, theta_center, ...)`
(radians, healpy's $(\theta, \phi)$ reversed). The candidate grid controls how
densely the patches overlap; `f_mask` and `f_mask_filter` how much masking a
patch tolerates in its 2PCF disc and in its aperture disc.
{func}`~pixpoint.select_patch_centers` does the selection on its own if you
want to inspect it.

## Pairs and apertures

```python
corr.preprocess()                   # pixel pairs inside every patch + aperture geometry
corr.save_pairs("patches.h5")       # reuse later with load_pairs
```

## Measuring

```python
res = corr.gg(shear_maps, w, flip_g1=True)        # M_a and xi_pm per patch
res.M_a.shape                                      # (nz, n_patches)
res.xi_p.shape                                     # (ncomb, n_patches, nbins)
res.gg_combinations, res.theta

res = corr.kk(delta_maps, w)                       # M_g and w(theta) per patch
res = corr.kg(delta_maps, shear_maps, w_lens, w_source, m_g=True)   # gamma_t (+ M_g)
res = corr.three_by_two_pt(shear_maps, delta_maps, weights=(w_source, w_lens))  # everything
zetas = res.zetas()                                # every i3PCF the measured fields form
```

An {class}`~pixpoint.IntegratedResult` holds the per-patch arrays; fields that
were not measured are `None`. `res.zetas()` averages the products over the
patches and returns a dictionary of the i3PCF variants, labelled by their
tomographic triplets (`Integrated3PCF.zeta_triplets(nz)`).

## Aperture filters

$M_{\rm ap}$ convolves the tangential shear about the centre with a compensated
filter $Q(\theta)$, by default Crittenden et al. (2002), which peaks at
$\sqrt 2\,\theta_Q$ and is truncated at $5\theta_Q$; Schneider et al. (1998)'s
polynomial filter is available as `Q_schneider`. $M_g$ is by default the
weighted mean of the scalar field over the patch disc, the top-hat window of
the patch 2PCFs (Halder et al. 2023); a filter $U$ such as `U_crittenden`
turns it into the aperture mass of a convergence map instead.

```python
from pixpoint import Q_schneider, U_crittenden

corr.preprocess(aperture_filter=Q_schneider)                  # Q for M_a
res = corr.kk(kappa_maps, w, aperture_filter=U_crittenden)    # M_g = M_ap of kappa
```

Any callable `f(theta, theta_Q)` with `theta` in radians and `theta_Q` in
arcminutes is accepted.

## Streaming a simulation suite

```python
from pixpoint import ZetaWriter

with ZetaWriter("zetas.h5", corr, flush_every=50) as out:
    for shear in maps:
        out.submit(corr.gg(shear, w, flip_g1=True))
```

{class}`~pixpoint.ZetaWriter` reduces every map's patches to its i3PCFs the
moment they are measured (9 kB per map instead of 1 MB of per-patch arrays)
and appends them to HDF5 from a background thread; `reduce="none"` keeps the
per-patch arrays for a jackknife. See [Files and streaming](io.md).

## Notes

- The i3PCF is a patch statistic: its covariance comes from the scatter
  between patches and realisations, and the patch set (centres, `patch_size`,
  `theta_Q`, `f_mask`) is part of the estimator definition. Store it with the
  data vector; the writer does.
- The [treecode](treecode.md) and a coarse `aperture_nside` apply here too
  and change the estimator in documented ways.
- Formal definitions: [estimator note](estimator_note.md).

## Reference

```{eval-rst}
.. autoclass:: pixpoint.Integrated3PCF
   :members: from_mask, preprocess, save_pairs, load_pairs, gg, kk, kg, three_by_two_pt, m_a, m_g, tomo_combinations, kg_combinations, zeta_triplets, zeta_cross_triplets, to_row_space, level_table
   :noindex:

.. autofunction:: pixpoint.select_patch_centers
   :noindex:
```

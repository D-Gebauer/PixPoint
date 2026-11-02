# Conventions

## Angles and units

- All angular scales you pass (`theta_min`, `theta_max`, `patch_size`,
  `theta_Q`, the scales of `OnePoint`) are in **arcminutes**; result bin
  centres (`theta`) are in arcminutes too. `binedges` are in radians.
- Patch centres of `Integrated3PCF` are `(phi, theta)` in **radians**, in
  healpy's convention: $\theta$ is the colatitude (0 at the north pole),
  $\phi$ the longitude. `healpy.pix2ang` returns `(theta, phi)`, so the
  argument order is reversed from what healpy gives you.
- Maps are HEALPix **RING** ordered.

## Shear sign and the position angle

Shear is projected onto the separation vector of a pair with the position
angle $\varphi$ measured at the pixel where the shear is defined, from the
local east direction towards north. The tangential and cross components are

$$
\gamma_{\rm t} = -\gamma_1 \cos 2\varphi - \gamma_2 \sin 2\varphi,
\qquad
\gamma_\times = \gamma_1 \sin 2\varphi - \gamma_2 \cos 2\varphi,
$$

so that a positive $\gamma_{\rm t}$ means tangential alignment about the other
pixel (a mass concentration produces positive $\gamma_{\rm t}$). In complex
notation $\gamma_{\rm t} + i\gamma_\times = -\gamma\, e^{-2i\varphi}$ with
$\gamma = \gamma_1 + i\gamma_2$. This is TreeCorr's convention on the sphere;
PixPoint agrees with TreeCorr's `GGCorrelation` and `KGCorrelation` on a
catalogue of the pixel centres to rounding.

Maps made with other conventions differ by the sign of one component, and
which one depends on how the map was made: healpy's polarisation convention
(its $\hat\theta$ axis points south) differs from the IAU convention (north
through east) by the sign of the second component, while a map binned from a
catalogue inherits whatever the catalogue's $e_1$, $e_2$ were defined with.
`flip_g1=True` / `flip_g2=True` on every shear method multiply that component
by −1 before measuring, at no cost, so you never have to rewrite a map.

Check the convention empirically rather than reasoning it out: measure
galaxy-galaxy lensing between the density contrast and the shear map of one
simulation (`Global2PCF.kg`), where $\gamma_{\rm t}$ must be positive on all
scales, and measure $\xi_\pm$ on an E-mode-only simulation, where a flipped
component turns $\xi_-$ into something that no longer follows the E-mode
prediction while $\xi_+$ is unaffected. Both checks take one call each.

## Which bin is which

Results carry their row labels; never infer them from the array shape.

- `gg`, `kk`: `combinations` lists the tomographic pairs $(i, j)$ with
  $i \le j$, row-major: (0,0), (0,1), ..., (0,nz−1), (1,1), ...
- `kg`: `combinations` lists (lens bin, source bin), lens-major.
- Cross rows sum both orientations of every pixel pair.
- Mixed samples (different numbers of lens and source bins) work everywhere.

The same orders without measuring:

```python
from pixpoint import Integrated3PCF
Integrated3PCF.tomo_combinations(4)                  # [(0,0), (0,1), ...]
Integrated3PCF.tomo_combinations(4, auto_only=True)  # the diagonal
Integrated3PCF.kg_combinations(2, 4)                 # (lens, source) rows for 2 lens, 4 source bins
```

## Three-point functions: the x-projection

{class}`~pixpoint.Global3PCF` reports spin-2 three-point functions in the
**x-projection**: a triangle is seen from its *centre* vertex, legs 2 and 3
run from the centre to the outer vertices, and the shear at every vertex is
projected along the leg that connects it to the centre (the centre's shear
along leg 2),

$$
\tilde\gamma = -\gamma\, e^{-2i\beta}, \qquad \mathrm{Re}\,\tilde\gamma = \gamma_{\rm t}
\text{ about the centre}.
$$

The opening angle is $\phi = \phi_3 - \phi_2$, the position angles of the two
legs as seen from the centre, and the multipoles are in $e^{in\phi}$. The four
shear components $\Gamma_0 \ldots \Gamma_3$ are labelled by which vertices are
conjugated: $\Gamma_0 = \langle \tilde\gamma\tilde\gamma\tilde\gamma\rangle$,
$\Gamma_1 = \langle \tilde\gamma^*\tilde\gamma\tilde\gamma\rangle$ (centre
conjugated), $\Gamma_2$ (leg 2), $\Gamma_3$ (leg 3). A theory prediction is
compared by converting it to this projection per triangle; the flat-sky
rotation to the centroid projection of Schneider & Lombardi (2003) is a
phase per triangle that you can apply yourself, and that PixPoint deliberately
does not apply for you.

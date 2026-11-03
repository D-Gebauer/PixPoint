# The PixPoint i3PCF estimator, and its static-treecode variant

A self-contained definition of what `Integrated3PCF` measures, written so that
it can be restated in a paper appendix. Version 8.0.0 (the footprint-wide
estimators `Global2PCF` and `Global3PCF` are defined on their documentation
pages and in `tests/threepoint_reference.py`). Companion measurements:
`benchmarks/static_treecode/STAGE1_RESULTS.md` (2-point level dependence),
`T10_RESULTS.md` (i3PCF level dependence), `BENCHMARK_TREECORR.md`
(cross-check against TreeCorr), `benchmarks/moments/RESULTS.md`
(moment-carrying cells).

Notation: pixels of a HEALPix map at resolution $N_{\rm side}$ are indexed by
$p$, with weights $w_p$ (e.g. the summed shear weight or galaxy count of the
pixel) and a binary footprint mask. Patches are discs of radius $R$ centred on
$\hat{n}_c$, $c = 1 \dots N_{\rm patch}$.

## 1. Patch-level quantities

**Aperture mass and aperture counts.** With a compensated filter $Q(\theta)$
(default: Crittenden et al. 2002), truncated at $5\theta_Q$, and $A_c$ the
solid angle of the patch,

$$ M_{\mathrm{ap},c} = A_c \frac{\sum_{p \in D_c} w_p\, \gamma_{\mathrm t}(p; \hat n_c)\, Q(\theta_{cp})}{\sum_{p \in D_c} w_p}, \qquad M_{\mathrm{g},c} = A_c \frac{\sum_{p \in D_c} w_p\, \delta_p\, Q(\theta_{cp})}{\sum_{p \in D_c} w_p}, $$

where $\gamma_{\rm t}(p;\hat n_c) = -\gamma_1 \cos 2\phi_{cp} - \gamma_2 \sin 2\phi_{cp}$
is the tangential shear about the patch centre and $D_c$ is the filter disc.

**Annulus 2-point functions.** For an angular bin $b$ with edges
$[\theta_b, \theta_{b+1})$, summing over ordered pairs $(p, q)$ of pixels that
both lie in patch $c$ and whose separation falls in bin $b$:

$$ \xi_{+,c}^{(b)} = \frac{\sum_{(p,q)} w_p w_q\, \tilde\gamma_p \tilde\gamma_q^*}{\sum_{(p,q)} w_p w_q}, \qquad \xi_{-,c}^{(b)} = \frac{\sum_{(p,q)} w_p w_q\, \tilde\gamma_p \tilde\gamma_q}{\sum_{(p,q)} w_p w_q}, \qquad \xi_{\mathrm g,c}^{(b)} = \frac{\sum_{(p,q)} w_p w_q\, \delta_p \delta_q}{\sum_{(p,q)} w_p w_q}, $$

where $\tilde\gamma_p = \gamma_p e^{-2i\varphi_p}$ is the complex shear rotated
into the frame of the pair separation ($\varphi_p$ is the position angle of $q$
as seen from $p$, and $\varphi_q$ that of $p$ as seen from $q$; the two
rotation factors are the per-pair geometry stored in the pair file), and

$$ \xi_{\mathrm t,c}^{(b)} = \frac{\sum_{(p,q)} w_p w_q\, \delta_p\, \gamma_{\mathrm t}(q; p)}{\sum_{(p,q)} w_p w_q} $$

for galaxy-galaxy lensing (lens pixel $p$, source pixel $q$).

**Tomography.** For a combination of tomographic bins $(i,j)$ with $i \ne j$
each pixel pair contributes in both orientations (bin $i$ at $p$ and $j$ at
$q$, and vice versa). PixPoint takes the **ratio of the summed orientations**,

$$ \xi^{(b),ij} = \frac{N^{(b)}_{ij} + N^{(b)}_{ji}}{W^{(b)}_{ij} + W^{(b)}_{ji}}, $$

i.e. the standard weighted estimator (TreeCorr's definition), not the mean of
the two orientation ratios. Since 5.0 this applies to $\xi_\pm$ as well as to
$\xi_{\rm g}$ and $\xi_{\rm t}$, and it applies to **every** entry point:
the private single-map references `_gg_single`, `_kk_single` and
`_kg_single` measure the same estimator as the vectorised and
fused paths. (`_kk_single` used the mean of the two orientation
ratios until 6.3.0; `tests/test_estimator_parity.py` is the gate.)

## 2. The i3PCF

For a triplet of tomographic bins $(z_c, z_2, z_3)$, the integrated 3-point
function is the patch-to-patch covariance of a central aperture quantity with
an annular 2-point function:

$$ \zeta^{(b)} = \frac{1}{N_{\rm patch}} \sum_c M_c\, \xi^{(b)}_c \;-\; \left( \frac{1}{N_{\rm patch}} \sum_c M_c \right) \left( \frac{1}{N_{\rm patch}} \sum_c \xi^{(b)}_c \right), $$

with $M \in \{M_{\rm ap}, M_{\rm g}\}$ and $\xi \in \{\xi_+, \xi_-, \xi_{\rm g},
\xi_{\rm t}\}$, giving the eight $\zeta$ variants of Halder et al. (2021)
(`zeta_a_plus`, `zeta_g_g`, …). The per-patch products are what
`calculate_all_zetas` averages, so the patch-to-patch scatter of
$M_c \xi^{(b)}_c$ is directly available as an internal error estimate.

## 3. The static treecode

The explicit pair geometry grows as $N_{\rm side}^4$: at $N_{\rm side} = 2048$
with $\theta_{\rm min} = 5'$ a DES-Y3-like patch set needs $\mathcal{O}(2\,
\mathrm{TB})$ of pair indices, which is the reason the estimator below exists.
The observation is that at large separations the pixel scale is far finer than
needed: a bin at $\theta \sim 100'$ does not resolve $1.7'$ pixels.

**Level assignment.** Given a resolution factor $k$, bin $b$ is measured at the
*coarsest* HEALPix resolution whose pixel size still resolves the bin,

$$ N_{\rm side}^{(b)} = \min \left\{ N \in \{1,2,4,\dots,N_{\rm side}\} : \; \mathrm{resol}(N) \le \theta_b / k \right\}, $$

with $\mathrm{resol}(N) = \sqrt{\Omega_{\rm pix}(N)}$ (healpy's `nside2resol`)
and $\theta_b$ the *lower* edge of the bin. The resulting table (one
$N_{\rm side}$ per bin, plus the realised $k_{\rm eff} = \theta_b /
\mathrm{resol}(N_{\rm side}^{(b)})$) is `Integrated3PCF.level_table`; it is stored
in the pair file and must be quoted with any measurement. $k$ is global: one
value for all statistics and all bins. `resolution_factor=None` (the default)
keeps every bin at the base resolution and reproduces the pre-5.0 estimator
bit-for-bit.

**Cells.** For each patch $c$ and each level $N$ present in the table, the
unmasked base pixels of that patch are grouped by their NESTED parent at
resolution $N$. Cells are built **per patch**, so a cell never straddles a
patch boundary and the top-hat patch window is preserved exactly. A cell $I$
carries

$$ w_I = \sum_{p \in I} w_p, \qquad \gamma_I = \frac{\sum_{p \in I} w_p \gamma_p}{w_I}, \qquad \delta_I = \frac{\sum_{p \in I} w_p \delta_p}{w_I}, $$

i.e. the weighted mean of its members (empty-weight cells are set to zero and
contribute nothing). Cell *positions* are the normalised mean unit vector of
the member pixel **centres**, computed from the binary mask only:

$$ \hat n_I = \frac{\sum_{p \in I} \hat n_p}{\left| \sum_{p \in I} \hat n_p \right|}. $$

Using the centroid rather than the HEALPix parent centre matters at the
1–3 % level on the coarsest levels and below 0.5 % for $k \ge 2$; using the
binary mask rather than the weights keeps the geometry independent of the maps,
so data and simulations share one pair file.

**Pair sums.** Bin $b$ is then evaluated exactly as in §1 but over the cells of
level $N_{\rm side}^{(b)}$: the pair is binned by the separation of the two
centroids, and the shear is rotated with the centroid-to-centroid angles.
Because $w_I \gamma_I = \sum_{p \in I} w_p \gamma_p$, a coarse pair sum is the
exact sum of its fine pairs' products, with one common bin assignment and one
common rotation instead of per-pair ones. Parallel transport of the member
shears to the centroid is not applied; it was measured to be $\le 10^{-4}$ even
for patches at $\mathrm{dec} = -62°$.

**Moment-carrying cells (8.0.0, `cell_moments`).** Given the bin, the only
approximation in a coarse pair sum of $\xi_-$ is the rotation phase: the child
pair $(p, q)$ has position angles that differ from the centroid pair's by
$\delta\phi_{pq}$, so its product carries $e^{2i\delta\phi_{pq}}$ relative to
the cell product (both ends turn by the same angle to first order, hence
$4\,\delta\phi$ for $\xi_-$ and $2\,\delta\phi$ for $\gamma_{\rm t}$, where
only the source end has a phase). Expanding that factor to second order in the
children's offsets $\delta r_p$ from the centroid (gnomonic, in the cell's
east/north frame) turns the cell pair into a contraction of per-cell moments of
the weighted shear,

$$ M_I = \sum_{p \in I} w_p \tilde\gamma_p, \qquad D_I = \sum_{p \in I} w_p \tilde\gamma_p\, \delta r_p, \qquad Q_I = \sum_{p \in I} w_p \tilde\gamma_p\, \delta r_p \delta r_p^{\mathsf T}, $$

with $\tilde\gamma_p$ the child shear parallel-transported into the cell frame
(exact geometry, from the binary mask like the centroids). With $d$ the pair
separation, $\hat p$, $\hat q$ the unit vectors perpendicular and parallel to
the connecting geodesic at each end (the $J$ end transported along it),
$D_p = \hat p \cdot D$, $Q_{pp} = \hat p^{\mathsf T} Q \hat p$ and so on, and
$S$ the sign of the rotation convention ($S = -1$ for PixPoint's
$e^{+2i\phi}$), the $\xi_-$ pair term is $\mathrm{Re}\,[F\, e^{-2iS(\phi_I + \phi_J)}]$ with

$$ F = M_I M_J - \frac{4iS}{d}\,(M_I D_{p,J} - D_{p,I} M_J) - \frac{8}{d^2}\,(M_I Q_{pp,J} - 2 D_{p,I} D_{p,J} + Q_{pp,I} M_J) + \frac{4iS}{d^2}\,(M_I Q_{qp,J} - D_{q,J} D_{p,I} - D_{q,I} D_{p,J} + Q_{qp,I} M_J). $$

$\gamma_{\rm t}$ has the same expansion with $4 \to 2$ and the lens cell's real
moments of $w\delta$; $\xi_+$ uses the transported $M$ only (its phase cancels);
$\xi_{\rm g}$ has no phase and no moments. `cell_moments="dipole"` keeps the
first-order term, `"quad"` (the default) both; `None` is the plain cell of §3.
The per-pair residual against the exact child sum scales as $4\delta\phi$,
$8\delta\phi^2$ and $(4\delta\phi)^3/6$ for the three choices (the dipole alone
removes the scatter but flips the sign of the mean bias, which only the
quadrupole removes). With `aperture_nside` the aperture sums of §1 expand the
same way about the coarse pixel centre, with the tangential phase and the
filter's derivatives $Q'$, $Q''$ at the cell (the patch-mean $M_{\rm g}$ has a
constant filter and needs nothing); full-resolution apertures have
$D = Q = 0$. The moments remove the *phase* window of the coarse cells; the
*binning* window (pairs whose fine separation falls in a neighbouring bin)
remains, see §4. `pixpoint.moments` holds the derivation and a NumPy
reference.

The aperture statistics of §1 are unaffected by $k$; they have their own
optional level (`aperture_nside`).

## 4. What the approximation does to the signal

The coarse estimator is a *differently windowed* statistic, not a noisier
version of the fine one: with realistic shape noise the per-patch coarse and
fine estimates have equal variance but are only partially correlated
($\rho \approx 0.65 / 0.83 / 0.93$ at $k_{\rm eff} \approx 2.1 / 3 / 6$).
Its effect on the signal is a suppression that depends only on $k_{\rm eff}$
(measured on Gaussian fields at $N_{\rm side} = 2048$, DES Y3 mask):

| $k_{\rm eff}$ | $\xi_-$ | $\gamma_{\rm t}$ | $\xi_+$, $\xi_{\rm g}$ |
|---|---|---|---|
| 1.45 | −29 % | −12 % | −2.5 % |
| 2.1 | −18…−20 % | −2…−6 % | ≲ 1 % |
| 3.0 | −8…−10 % | −2…−2.6 % | ≲ 1 % |
| 4.3 | −4.3…−5.3 % | −0.8…−2.4 % | < 0.7 % |
| 6.1 | −2.0…−2.9 % | −0.3…−0.5 % | < 0.3 % |
| 8.7 | −0.9…−1.1 % | ≤ 0.3 % | < 0.3 % |

At the i3PCF level, on real N-body shear maps with the DES Y3 mask at
$N_{\rm side} = 512$ and 917 patches, $k = 2.9$ shifts every entry of the
$\zeta_{a,\pm}$ data vector by less than $0.3\sigma$ of the patch-to-patch
error of one measurement (0.6 / 1.0 $\sigma$ in quadrature over the 140-entry
vectors), while $k = 1.5$ produces a coherent 20–40 % suppression of
$\zeta_{a,-}$ (up to $0.8\sigma$). $\xi_-$ / $\zeta_{a,-}$ sets the limit;
$k \gtrsim 3$ is safe, $k \lesssim 1.5$ is not.

The ratio of coarse to fine is a smooth function of $k_{\rm eff}$ that is
identical for weight distributions typical of data and of simulations
(agreement to $10^{-3}$), so the choice of $k$ does not bias a
simulation-based inference **as long as the same $k$ is used throughout**.
Coarse and fine data vectors differ at the noise level, not only by the
few-per-cent signal suppression, and must never be mixed between data,
simulations and covariance.

**With moment-carrying cells** (`cell_moments="quad"`, the 8.0.0 default;
the table above is the plain cell) the phase part of the suppression goes:
on the test field at $k = 4$ the mean $\xi_-$ error drops from −2…−5 % to
$\le 0.3$ % and the information retained under DES-like shape noise from
0.97–0.99 to $\ge 0.9999$; $\gamma_{\rm t}$ follows with half the phase,
$\xi_+$ moves at the $10^{-5}$ level (transported frame). At the i3PCF level
(same T10 setup) the $\zeta_{a,-}$ price of $k = 2.9$ falls from 0.27 to
0.20 $\sigma_{\rm patch}$ and that of $k = 1.5$ from 0.81 to 0.39; $k = 2$
with quad (0.39) does not pass the criterion $k = 2.9$ passes, so $k \approx 3$
remains the recommendation. What is left is the binning window, pairs
exchanged across bin edges (±2 % at $k = 4$, alternating in sign at the level
seams), which $\zeta_{a,+}$ already showed and which the moments cannot touch.
It is deterministic and shared by data and simulations; a smooth data vector,
or one comparable with an exact code, means full resolution.


## 5. One-point statistics (`OnePoint`, 8.0.0)

The same aperture quantities as in §1, at centres $c$ that tile the footprint instead of a
patch set: every pixel of a HEALPix grid at $N_{\rm side}^{\rm c}$ (pixel $\approx \theta/2$
by default) whose disc *coverage*

$$ f_c = \frac{|\{p \in \mathcal D_c : p\ \text{unmasked}\}|}{|\mathcal D_c|} $$

(binary mask at $N_{\rm side}$, independent of the weights and of the aperture window) is at
least $f_{\min}$ (0.8). At each accepted centre, for a scale $\theta$,

$$ M_{\mathrm{ap},c} = A_c \frac{\sum_{p\in D_c} w_p\, \gamma_{\rm t}(p;\hat n_c)\, Q(\theta_{cp})}{\sum_{p \in D_c} w_p}, \qquad
   M_{\times,c} = A_c \frac{\sum_{p\in D_c} w_p\, \gamma_\times(p;\hat n_c)\, Q(\theta_{cp})}{\sum_{p \in D_c} w_p}, $$

with $\gamma_\times = \gamma_1 \sin 2\phi - \gamma_2 \cos 2\phi$, $D_c$ the unmasked pixels within
$5\theta$ (where $Q$ with $\theta_Q = \theta$ has decayed to $10^{-3}$) and $A_c$ their solid
angle -- exactly $M_{\mathrm{ap}}$ of §1 with $R = \theta_Q = \theta$; and for a scalar field
$f$ (κ or δ) one of

$$ k_c = \frac{\sum_{p \in T_c} w_p f_p}{\sum_{p\in T_c} w_p} \;\;(T_c: \text{disc of radius } \theta), \qquad
   k_c = \frac{\sum_{p \in c} w_p f_p}{\sum_{p \in c} w_p} \;\;(\text{the pixel } c \text{ of } N_{\rm side}^{\rm c}), \qquad
   k_c = A_c \frac{\sum_{p\in D_c} w_p f_p U(\theta_{cp})}{\sum_{p\in D_c} w_p}. $$

With `aperture_nside` the sums run over the cells of the degraded map (weighted means, moment
corrections as for $M_{\rm ap}$), the coverage does not change.

The estimator reports, per scale and tomographic bin, sums over the accepted apertures with
**unit weight** (the DES Y3 moments convention; the aperture value is already normalised by its
$\sum w$): the histogram on fixed edges $[e_b, e_{b+1})$ with under- and overflow, the raw
moments $\sum_c x_c^k$ for $k = 1\dots5$, and the tomographic cross moments
$\sum_c x_{i,c}\, x_{j,c}$, $\sum_c x_{i,c}^2 x_{j,c}$, $\sum_c x_{i,c} x_{j,c} x_{k,c}$. An aperture
whose disc holds no weight in one of the tomographic bins ($\sum_{p} w_p = 0$ there) is left out
of all sums; the count of apertures that entered is reported. PDF and moments are the ratios.

Caveats for an analysis: the centres overlap heavily (the aperture is $5\theta$ wide, the
centre spacing $\theta/2$), so the histogram counts are not independent and the PDF's
covariance must come from simulations measured with the same centres; the coverage cut is a
mask-dependent selection that data and simulations must share (same mask, same $f_{\min}$,
same `centre_nside`); and `aperture_nside` is a window on the field that moves the tails of
the PDF -- quantify it against full resolution before using it.

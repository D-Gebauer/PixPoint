"""One-point statistics of aperture fields on HEALPix maps: :class:`OnePoint`.

The pair estimators of this package measure correlations; ``OnePoint``
measures the *distribution* of an aperture field over the footprint -- the
aperture-mass PDF of a shear map (M_ap and its cross component M_x), the PDF
of a convergence or density map smoothed with a top hat or a compensated
filter, and their moments -- with the same row space, map inputs, weights,
backends and ``aperture_nside`` window as
:class:`~pixpoint.correlations.Integrated3PCF`.

**Estimator.**  For a scale θ the centres are the HEALPix pixels of
``centre_nside`` (default: the level whose pixel is closest to θ/2) whose
coarse pixel holds an unmasked map pixel and whose aperture disc has a
*coverage* -- the fraction of its pixels that are unmasked, from the binary
mask, independent of the map weights -- of at least ``min_coverage``.  At
every accepted centre c (``A_c`` the disc area, ``D_c`` its unmasked pixels):

* ``g``: ``M_ap,c = A_c Σ_{p∈D_c} w_p γ_t(p;c) Q(θ_cp) / Σ w_p`` and
  ``M_x,c`` with ``γ_x = γ_1 sin 2φ − γ_2 cos 2φ`` in place of ``γ_t``; disc
  radius ``radius_factor · θ`` (5 θ by default, the support of ``Q_crittenden``);
* ``k`` with ``k_aperture_filter=None``: ``k_c = Σ w_p f_p / Σ w_p`` over the
  disc of radius θ (circular top hat);
* ``k`` with a filter U: ``k_c = A_c Σ w_p f_p U(θ_cp) / Σ w_p`` over the
  ``radius_factor · θ`` disc (e.g. ``U_crittenden``: the aperture mass of a
  convergence map);
* ``k`` with ``"healpix"``: ``k_c = Σ_{p ∈ pixel c} w_p f_p / Σ w_p`` over the
  unmasked children of the centre pixel (the HEALPix-pixel top hat, i.e. the
  masked ``ud_grade`` of the map).

These are the ``M_a`` and ``M_g`` of ``Integrated3PCF`` with the centres as
patches -- the same kernels, bitwise -- and with ``aperture_nside`` they are
measured on the degraded map (moment-carrying cells included).  ``M_x`` is
the same kernel on the swapped components ``(γ_2, −γ_1)``, so it is exact
by construction and adds one pass over the disc geometry.

Per scale and tomographic bin the values are reduced to a histogram on
fixed edges (half-open bins, under- and overflow reported), the raw moments
``Σ x^k`` (k = 1..5) and the tomographic cross moments ``Σ x_i x_j``,
``Σ x_i² x_j``, ``Σ x_i x_j x_k`` -- every accepted aperture with **unit
weight** (the DES Y3 moments convention; the aperture value is already
normalised by its own ``Σ w``).  An aperture whose disc holds no weight in
one of the tomographic bins (``Σ w = 0`` there; found from the weights
alone, one extra pass over the discs per call, since the kernels return NaN
or 0 for it depending on the backend) is left out of the sums and the
histogram of every bin; ``count`` says how many entered.  The reduction runs on
the backend's array module; the sums are returned on the host and add
across devices and footprints (:class:`~pixpoint.results.OnePointResult`).

**Not here: mass mapping.**  κ from shear needs spherical harmonic
transforms; produce κ (or δ) upstream and pass it to ``k``.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import healpy as hp
import numpy as np
from tqdm import trange

from .correlation_helpers import _xp
from .correlations import Integrated3PCF
from .multi_device import (
    balanced_ranges,
    device_list,
    require_host_arrays,
    run_on_devices,
)
from .pair_geometry import KApertureGeometry, PairGeometry
from .results import ONEPOINT_POWERS, OnePointResult, onepoint_cross_index
from .treecode import is_power_of_two

# Geometry kinds an engine set is built for: the shear discs (``g`` and ``k``
# with a filter U), the circular top hat of radius θ, the HEALPix pixel.
_KINDS = ("shear", "tophat", "pixel")


def default_centre_nside(scale_arcmin: float, nside: int) -> int:
    """The power-of-two nside (at most ``nside``) whose pixel size is closest,
    in log, to ``scale_arcmin / 2``."""
    target = np.log(scale_arcmin / 2.0)
    best, best_dist = 1, np.inf
    n = 1
    while n <= nside:
        dist = abs(np.log(hp.nside2resol(n, arcmin=True)) - target)
        if dist < best_dist:
            best, best_dist = n, dist
        n *= 2
    return best


def footprint_centres(mask: np.ndarray, nside: int, centre_nside: int) -> np.ndarray:
    """RING ids (ascending) of the ``centre_nside`` pixels with at least one
    unmasked child at ``nside``."""
    shift = 2 * (int(np.log2(nside)) - int(np.log2(centre_nside)))
    nest = hp.ring2nest(nside, np.flatnonzero(mask).astype(np.int64))
    return np.unique(hp.nest2ring(centre_nside, nest >> shift)).astype(np.int64)


def _parents(nside_fine: int, pix: np.ndarray, nside_coarse: int) -> np.ndarray:
    shift = 2 * (int(np.log2(nside_fine)) - int(np.log2(nside_coarse)))
    nest = hp.ring2nest(nside_fine, np.asarray(pix, dtype=np.int64))
    return np.asarray(hp.nest2ring(nside_coarse, nest >> shift), dtype=np.int64)


def _to_host(a: Any) -> np.ndarray:
    if hasattr(a, "get") and not isinstance(a, np.ndarray):
        return np.asarray(a.get())
    return np.asarray(a)


def _filter_name(f: Any) -> Any:
    if f is None or isinstance(f, str):
        return f
    return getattr(f, "__name__", repr(f))


def _unit_filter(theta: Any, theta_Q: float = 0.0) -> Any:
    """The constant filter 1 on the aperture discs (the Σw pass)."""
    del theta_Q
    return np.ones_like(np.asarray(theta, dtype=np.float64))


def _nonempty_apertures(engine: "_ApertureEngine", kind: str, w: Any) -> Any:
    """``(n_centres,)`` bool: apertures whose disc holds weight ``Σ w > 0`` in
    *every* tomographic bin.

    The aperture kernels disagree on an empty disc -- NaN from the Numba
    shear kernel, 0 from the GPU ``safe_divide`` and from every ``M_g`` path
    -- and a 0 would silently enter the PDF, so the test is made here from
    the weights alone: ``m_g`` with the weights as the field, unit weights and
    the unit filter gives ``A Σ w / N`` per disc, which is 0 exactly when every
    weight is (the weights are non-negative).  One extra pass over the disc
    geometry per call, no map involved.
    """
    xp = np if isinstance(w, np.ndarray) else engine.backend.module
    filt = _unit_filter if kind == "shear" else None
    sw = engine.m_g(w, xp.ones_like(w), aperture_filter=filt, return_device=True)
    return xp.all(sw > 0, axis=0)


class _ApertureEngine(Integrated3PCF):
    """An ``Integrated3PCF`` whose patches are aperture centres and which has
    no pairs: only the aperture statistics are ever called."""

    _pixel_tophat: Optional[KApertureGeometry] = None

    def find_pairs(self) -> None:  # pragma: no cover - guarded by OnePoint
        raise RuntimeError("a OnePoint engine has no pair geometry")

    def _k_aperture_geometry(
        self, aperture_filter: Optional[Callable[..., Any]] = None
    ) -> KApertureGeometry:
        if aperture_filter is None and self._pixel_tophat is not None:
            return self._pixel_tophat
        return super()._k_aperture_geometry(aperture_filter)


def _pixel_tophat_geometry(
    engine: Integrated3PCF, centre_nside: int, centre_pix: np.ndarray
) -> KApertureGeometry:
    """Row-space CSR of the HEALPix-pixel top hat: the members of every
    centre pixel (unmasked pixels, or the observed cells of
    ``aperture_nside``), unit filter values and unit area."""
    npix = engine.npix
    ap_cells = engine._aperture_cells()
    if ap_cells is None:
        nside_disc = int(engine.nside)
        member_pix = np.asarray(engine.map_inds, dtype=np.int64)
        member_ids = member_pix
    else:
        assert engine.aperture_nside is not None
        nside_disc = int(engine.aperture_nside)
        member_pix = np.asarray(ap_cells[0], dtype=np.int64)
        member_ids = npix + np.arange(member_pix.size, dtype=np.int64)
    centre_pix = np.asarray(centre_pix, dtype=np.int64)
    parent = _parents(nside_disc, member_pix, centre_nside)
    pos = np.searchsorted(centre_pix, parent)
    pos = np.minimum(pos, max(centre_pix.size - 1, 0))
    hit = (centre_pix[pos] == parent) if centre_pix.size else np.zeros(0, bool)
    pos = pos[hit]
    ids = member_ids[hit]
    order = np.argsort(pos, kind="stable")
    counts = np.bincount(pos, minlength=centre_pix.size)
    offsets = np.zeros(centre_pix.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    rows = engine._global_ids_to_rows(
        ids[order].astype(engine.index_dtype), engine._global_to_row_lut()
    )
    rot = engine.rotation_dtype
    return KApertureGeometry(
        np.ascontiguousarray(rows, dtype=engine.index_dtype),
        np.ones(rows.size, dtype=rot),
        offsets,
        np.ones(centre_pix.size, dtype=rot),
    )


def _stack0(xp: Any, items: List[Any]) -> Any:
    """A 1-D float64 array from 0-d values (host or device); empty if none."""
    if not items:
        return xp.zeros(0, dtype=xp.float64)
    return xp.stack([xp.asarray(x, dtype=xp.float64) for x in items])


def _histogram(
    xp: Any, v: Any, edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts of ``v (nz, n)`` on the half-open bins of ``edges (nz, ne)``:
    ``(hist (nz, nbins), underflow (nz,), overflow (nz,))``, host float64."""
    nz, ne = edges.shape
    nbins = ne - 1
    hist = np.zeros((nz, nbins), dtype=np.float64)
    under = np.zeros(nz, dtype=np.float64)
    over = np.zeros(nz, dtype=np.float64)
    for z in range(nz):
        idx = xp.searchsorted(xp.asarray(edges[z]), v[z], side="right") - 1
        under[z] = float(_to_host(xp.count_nonzero(idx < 0)))
        over[z] = float(_to_host(xp.count_nonzero(idx >= nbins)))
        ok = (idx >= 0) & (idx < nbins)
        counts = xp.bincount(idx[ok], minlength=nbins)
        hist[z] = _to_host(counts)[:nbins]
    return hist, under, over


def _reduce_scale(
    vals: Dict[str, Any],
    valid: Any,
    names: Sequence[str],
    edges: Optional[np.ndarray],
    keep_values: bool,
) -> Dict[str, Any]:
    """Sums of one scale on one device: ``vals[name]`` is ``(nz, n)`` on the
    host or the device, ``valid`` the ``(n,)`` mask of apertures with weight
    in every bin.  Everything returned is host float64."""
    first = vals[names[0]]
    xp = _xp(first)
    nz, n = int(first.shape[0]), int(first.shape[1])
    index = onepoint_cross_index(nz)
    out: Dict[str, Any] = {"n": n, "fields": {}}
    valid = xp.asarray(valid, dtype=bool)
    for name in names:
        raw = vals[name]
        v = xp.asarray(raw, dtype=xp.float64)
        keep = valid & xp.all(xp.isfinite(v), axis=0)
        v = v[:, keep]
        count = int(v.shape[1])
        powers = xp.stack(
            [xp.sum(xp.power(v, k), axis=1) for k in range(1, ONEPOINT_POWERS + 1)],
            axis=1,
        )
        v2 = v * v
        gram = v @ v.T  # Σ x_i x_j
        gram2 = v2 @ v.T  # Σ x_i² x_j
        cross = {
            "ij": _stack0(xp, [gram[i, j] for i, j in index["ij"]]),
            "iij": _stack0(xp, [gram2[i, j] for i, j in index["iij"]]),
            "ijk": _stack0(
                xp, [xp.sum(v[i] * v[j] * v[k]) for i, j, k in index["ijk"]]
            ),
        }
        field: Dict[str, Any] = {
            "count": count,
            "power": _to_host(powers),
            "cross": {k: _to_host(c) for k, c in cross.items()},
        }
        if edges is not None:
            field["hist"], field["under"], field["over"] = _histogram(xp, v, edges)
        if keep_values:
            field["values"] = _to_host(raw).astype(np.float64, copy=False)
        out["fields"][name] = field
    return out


def _add_scale(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": a["n"] + b["n"], "fields": {}}
    for name, fa in a["fields"].items():
        fb = b["fields"][name]
        f: Dict[str, Any] = {
            "count": fa["count"] + fb["count"],
            "power": fa["power"] + fb["power"],
            "cross": {k: fa["cross"][k] + fb["cross"][k] for k in fa["cross"]},
        }
        if "hist" in fa:
            for k in ("hist", "under", "over"):
                f[k] = fa[k] + fb[k]
        if "values" in fa:
            f["values"] = np.concatenate((fa["values"], fb["values"]), axis=1)
        out["fields"][name] = f
    return out


class OnePoint:
    """One-point statistics (PDF and moments) of aperture fields.

    Args:
        nside: HEALPix resolution of the maps (power of two).
        mask: HEALPix map at ``nside`` (nonzero = observed); ``None`` for
            the full sky.  Defines the row space, as for the other
            estimators, and the coverage of every aperture.
        scales: Aperture scales θ in arcminutes, one measurement per scale.
        centre_nside: HEALPix level of the aperture centres, one value or one
            per scale.  Default per scale: the level whose pixel is closest
            to θ/2.
        aperture_nside: As for ``Integrated3PCF``: ``None`` measures the
            apertures on the map pixels, a coarser power of two on the
            degraded map (same window, same moment-carrying cells).
        min_coverage: Minimum fraction of an aperture's disc that must be
            unmasked (binary mask at ``nside``, whatever ``aperture_nside``
            is, so that the window does not move the centre set) for the
            centre to be accepted; 0 accepts every footprint centre whose
            disc holds at least one unmasked pixel.
        aperture_filter: Q of the aperture mass (``g``); default
            ``Q_crittenden``.
        k_aperture_filter: What ``k`` measures: ``None`` (default) the
            circular top hat of radius θ, ``"healpix"`` the HEALPix-pixel
            top hat of the centre pixel, or a filter U (``U_crittenden``,
            ``U_schneider``, any ``f(theta, theta_Q)``) on the ``g`` discs.
        radius_factor: Disc radius of the filtered statistics in units of
            θ (5: where ``Q_crittenden`` has decayed to 1e-3 of its peak;
            ``Q_schneider``/``U_schneider`` vanish beyond 1).
        edges: Histogram edges, fixed up front (from the simulations):
            one ascending array for every scale and bin, or ``(nscale,
            nedges)``.  ``None`` measures moments only; a call may pass its
            own ``edges`` (also ``(nz, nscale, nedges)``).
        device: ``"cpu"``, ``"gpu"``, ``"auto"``, a GPU id, or a list of GPU
            ids.  A list splits the centres of every scale across the
            devices (contiguous ranges of about equal disc pixel count);
            measurements then take host arrays, each device reduces its
            centres and the sums add.
        map_precision, rotation_precision, cell_moments, fastmath: as for
            ``Integrated3PCF``.

    The geometry of a statistic is built on first use (or by
    :meth:`prepare`): the centre selection, then the aperture discs (``g``
    and ``k`` with a filter), the top-hat discs or the pixel members.
    """

    def __init__(
        self,
        nside: int,
        mask: Optional[np.ndarray] = None,
        scales: Sequence[float] = (20.0, 45.0, 90.0),
        *,
        centre_nside: Union[None, int, Sequence[int]] = None,
        aperture_nside: Optional[int] = None,
        min_coverage: float = 0.8,
        aperture_filter: Optional[Callable[..., Any]] = None,
        k_aperture_filter: Union[None, str, Callable[..., Any]] = None,
        radius_factor: float = 5.0,
        edges: Optional[Any] = None,
        device: Union[str, int, Sequence[Union[str, int]]] = "auto",
        map_precision: Union[str, np.dtype, type] = "float64",
        rotation_precision: Union[str, np.dtype, type] = "float32",
        cell_moments: Optional[str] = "quad",
        fastmath: bool = True,
    ) -> None:
        nside = int(nside)
        if nside <= 0 or not is_power_of_two(nside):
            raise ValueError(f"nside must be a power of two; got {nside}")
        self.nside = nside
        self.npix = int(hp.nside2npix(nside))
        self.scales = np.atleast_1d(np.asarray(scales, dtype=np.float64))
        if self.scales.ndim != 1 or self.scales.size == 0 or np.any(self.scales <= 0):
            raise ValueError("scales must be a non-empty sequence of positive arcmin")
        if aperture_nside is not None:
            aperture_nside = int(aperture_nside)
            if not is_power_of_two(aperture_nside) or aperture_nside > nside:
                raise ValueError(
                    "aperture_nside must be a power of two <= nside; got "
                    f"{aperture_nside}"
                )
            if aperture_nside == nside:
                aperture_nside = None
        self.aperture_nside = aperture_nside
        if centre_nside is None:
            cn = [default_centre_nside(float(t), nside) for t in self.scales]
        elif isinstance(centre_nside, (int, np.integer)):
            cn = [int(centre_nside)] * self.nscale
        else:
            cn = [int(c) for c in centre_nside]
            if len(cn) != self.nscale:
                raise ValueError("centre_nside needs one value per scale")
        for c in cn:
            if c <= 0 or not is_power_of_two(c) or c > nside:
                raise ValueError(
                    f"centre_nside must be a power of two <= nside; got {c}"
                )
        self.centre_nside: Tuple[int, ...] = tuple(cn)
        if not (0.0 <= float(min_coverage) <= 1.0):
            raise ValueError("min_coverage must lie in [0, 1]")
        self.min_coverage = float(min_coverage)
        self.aperture_filter = aperture_filter
        if k_aperture_filter is not None and not (
            k_aperture_filter == "healpix" or callable(k_aperture_filter)
        ):
            raise ValueError(
                "k_aperture_filter must be None (top hat), 'healpix' or a "
                f"filter callable; got {k_aperture_filter!r}"
            )
        self.k_aperture_filter = k_aperture_filter
        if not (float(radius_factor) > 0):
            raise ValueError("radius_factor must be positive")
        self.radius_factor = float(radius_factor)
        self.edges: Optional[np.ndarray] = None
        if edges is not None:
            e = np.asarray(edges, dtype=np.float64)
            if e.ndim not in (1, 2) or (e.ndim == 2 and e.shape[0] != self.nscale):
                raise ValueError(
                    "edges must be 1-D (every scale) or (nscale, nedges); "
                    f"got shape {e.shape}"
                )
            self._check_edges(e)
            self.edges = e
        self.devices: List[Any] = (
            device_list(device) if isinstance(device, (list, tuple, set)) else [device]
        )
        self.map_precision = map_precision
        self.rotation_precision = rotation_precision
        self.cell_moments = cell_moments
        self.fastmath = fastmath

        if mask is None:
            self._mask = np.ones(self.npix, dtype=bool)
            self._mask_arg: Optional[np.ndarray] = None
        else:
            mask = np.asarray(mask)
            if mask.shape != (self.npix,):
                raise ValueError("mask must be a HEALPix map at nside")
            self._mask = mask != 0
            self._mask_arg = self._mask
        # kind -> scale index -> engines (one per device) / centre selection
        self._engines: Dict[str, Dict[int, List[_ApertureEngine]]] = {
            k: {} for k in _KINDS
        }
        self._centres: Dict[str, Dict[int, Dict[str, Any]]] = {k: {} for k in _KINDS}

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #

    @property
    def nscale(self) -> int:
        return int(self.scales.size)

    @property
    def _k_kind(self) -> str:
        if self.k_aperture_filter is None:
            return "tophat"
        if self.k_aperture_filter == "healpix":
            return "pixel"
        return "shear"

    def _kind_of(self, statistic: str) -> str:
        if statistic == "g":
            return "shear"
        if statistic == "k":
            return self._k_kind
        raise ValueError(f"statistic must be 'g' or 'k'; got {statistic!r}")

    def centres(self, statistic: str, scale_index: int) -> Dict[str, Any]:
        """The accepted centres of ``statistic`` at scale ``scale_index``:
        ``centre_nside``, ``pix`` (RING ids), ``theta``/``phi`` (radians),
        ``coverage`` and ``disc_size`` (observed pixels) per centre,
        ``n_candidates`` footprint centres before the coverage cut."""
        kind = self._kind_of(statistic)
        self._parts(kind, scale_index)
        sel = self._centres[kind][scale_index]
        theta, phi = hp.pix2ang(sel["centre_nside"], sel["pix"])
        return {
            "centre_nside": sel["centre_nside"],
            "pix": sel["pix"].copy(),
            "theta": np.asarray(theta),
            "phi": np.asarray(phi),
            "coverage": sel["coverage"].copy(),
            "disc_size": sel["disc_size"].copy(),
            "n_candidates": int(sel["n_candidates"]),
        }

    def n_apertures(self, statistic: str) -> np.ndarray:
        """Accepted centres of ``statistic`` per scale, ``(nscale,)``."""
        kind = self._kind_of(statistic)
        for s in range(self.nscale):
            self._parts(kind, s)
        return np.array(
            [self._n_centres(kind, s) for s in range(self.nscale)], dtype=np.int64
        )

    def _n_centres(self, kind: str, s: int) -> int:
        return int(self._centres[kind][s]["pix"].size)

    @property
    def provenance(self) -> Dict[str, Any]:
        """What this estimator measures (store it next to every result)."""
        return {
            "estimator": "OnePoint",
            "nside": self.nside,
            "aperture_nside": self.aperture_nside,
            "scales_arcmin": [float(t) for t in self.scales],
            "centre_nside": list(self.centre_nside),
            "min_coverage": self.min_coverage,
            "radius_factor": self.radius_factor,
            "aperture_filter": _filter_name(
                PairGeometry.resolve_aperture_filter(self.aperture_filter)
            ),
            "k_aperture_filter": _filter_name(self.k_aperture_filter),
            "cell_moments": self.cell_moments,
            "map_precision": str(np.dtype(self.map_precision)),
            "rotation_precision": str(np.dtype(self.rotation_precision)),
            "n_apertures": {
                kind: [self._n_centres(kind, s) for s in sorted(sel)]
                for kind, sel in self._centres.items()
                if sel
            },
        }

    def __repr__(self) -> str:
        return (
            f"<OnePoint nside={self.nside} scales={list(self.scales)} "
            f"centre_nside={list(self.centre_nside)} devices={self.devices}>"
        )

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    def prepare(self, statistics: Sequence[str] = ("g", "k")) -> None:
        """Build the centres and aperture geometry of ``statistics`` for every
        scale now (otherwise it happens on first use)."""
        for statistic in statistics:
            kind = self._kind_of(statistic)
            for s in range(self.nscale):
                self._parts(kind, s)

    def release_device_memory(self) -> None:
        """Drop the device buffers of every engine (host geometry stays)."""
        for per_scale in self._engines.values():
            for parts in per_scale.values():
                for eng in parts:
                    eng.release_device_memory()

    def _disc_radius_arcmin(self, kind: str, s: int) -> float:
        theta = float(self.scales[s])
        return theta if kind == "tophat" else self.radius_factor * theta

    def _select_centres(self, kind: str, s: int) -> Dict[str, Any]:
        """Footprint centres of scale ``s`` with the coverage of their disc.

        Coverage is the unmasked fraction of the disc's *map* pixels (at
        ``nside``), whatever ``aperture_nside`` is: the window of the
        statistic must not move the centre set, or two windows could not be
        compared on the same apertures.
        """
        cn = self.centre_nside[s]
        cand = footprint_centres(self._mask, self.nside, cn)
        if kind == "pixel":
            nside_disc = self.aperture_nside or self.nside
            if cn > nside_disc:
                raise ValueError(
                    "the HEALPix-pixel top hat needs centre_nside <= aperture_nside "
                    f"(got {cn} > {nside_disc})"
                )
            obs_pix = np.flatnonzero(self._mask)
            pos = np.searchsorted(cand, _parents(self.nside, obs_pix, cn))
            sizes = np.bincount(pos, minlength=cand.size).astype(np.int64)
            n_child = 4 ** (int(np.log2(self.nside)) - int(np.log2(cn)))
            coverage = sizes / float(n_child)
        else:
            radius = np.radians(self._disc_radius_arcmin(kind, s) / 60.0)
            vecs = np.asarray(hp.pix2vec(cn, cand)).T
            sizes = np.zeros(cand.size, dtype=np.int64)
            coverage = np.zeros(cand.size, dtype=np.float64)
            for i in trange(
                cand.size,
                desc=f"OnePoint centres {self.scales[s]:g}'",
                unit=" centres",
            ):
                disc = hp.query_disc(self.nside, vecs[i], radius)
                k = int(np.count_nonzero(self._mask[disc]))
                sizes[i] = k
                coverage[i] = k / disc.size if disc.size else 0.0
        accepted = (coverage >= self.min_coverage) & (sizes > 0)
        return {
            "centre_nside": cn,
            "n_candidates": int(cand.size),
            "pix": cand[accepted],
            "coverage": coverage[accepted],
            "disc_size": sizes[accepted],
        }

    def _parts(self, kind: str, s: int) -> List[_ApertureEngine]:
        parts = self._engines[kind].get(s)
        if parts is not None:
            return parts
        sel = self._select_centres(kind, s)
        n = int(sel["pix"].size)
        theta_s = float(self.scales[s])
        if n == 0:
            raise ValueError(
                f"no centre at {theta_s:g}' reaches min_coverage="
                f"{self.min_coverage}; loosen it or use a finer centre_nside"
            )
        if n < len(self.devices):
            raise ValueError(
                f"{n} centres at {theta_s:g}' cannot be split across "
                f"{len(self.devices)} devices"
            )
        theta_c, phi_c = hp.pix2ang(sel["centre_nside"], sel["pix"])
        ranges = (
            balanced_ranges(sel["disc_size"], len(self.devices))
            if len(self.devices) > 1
            else [(0, n)]
        )
        parts = []
        for dev, (a, b) in zip(self.devices, ranges):
            eng = _ApertureEngine(
                self.nside,
                np.asarray(phi_c[a:b], dtype=np.float64),
                np.asarray(theta_c[a:b], dtype=np.float64),
                nbins=1,
                theta_min=1.0,
                theta_max=2.0,
                patch_size=theta_s,
                theta_Q=theta_s,
                mask=self._mask_arg,
                fastmath=self.fastmath,
                device=dev,
                map_precision=self.map_precision,
                rotation_precision=self.rotation_precision,
                resolution_factor=None,
                aperture_nside=self.aperture_nside,
                cell_moments=self.cell_moments,
            )
            # the single source of the disc radius (pair_geometry reads it)
            eng.radius_filter = self.radius_factor * theta_s
            if kind == "shear":
                eng.find_aperture_pairs(aperture_filter=self.aperture_filter)
            elif kind == "pixel":
                eng._pixel_tophat = _pixel_tophat_geometry(
                    eng, sel["centre_nside"], sel["pix"][a:b]
                )
            parts.append(eng)
        sel["ranges"] = ranges
        self._centres[kind][s] = sel
        self._engines[kind][s] = parts
        return parts

    # ------------------------------------------------------------------ #
    # measurement
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_edges(e: np.ndarray) -> None:
        if e.shape[-1] < 2 or not np.all(np.diff(e, axis=-1) > 0):
            raise ValueError("edges must hold >= 2 strictly increasing values")

    def _resolve_edges(self, edges: Optional[Any], nz: int) -> Optional[np.ndarray]:
        e_in = self.edges if edges is None else edges
        if e_in is None:
            return None
        e = np.asarray(e_in, dtype=np.float64)
        if e.ndim == 1:
            e = np.broadcast_to(e, (nz, self.nscale, e.size))
        elif e.ndim == 2 and e.shape[0] == self.nscale:
            e = np.broadcast_to(e[None], (nz, self.nscale, e.shape[1]))
        elif e.ndim == 3 and e.shape[:2] == (nz, self.nscale):
            pass
        else:
            raise ValueError(
                "edges must be 1-D, (nscale, nedges) or (nz, nscale, nedges); "
                f"got shape {e.shape} for nz={nz}, nscale={self.nscale}"
            )
        self._check_edges(e)
        return np.ascontiguousarray(e)

    @staticmethod
    def _nz(w: Any) -> int:
        shape = getattr(w, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError("weights must be (nz, npix | n_active)")
        return int(shape[0])

    def _measure(
        self,
        statistic: str,
        kind: str,
        names: Tuple[str, ...],
        measure: Callable[[_ApertureEngine], Dict[str, Any]],
        inputs: Sequence[Any],
        w: Any,
        edges: Optional[Any],
        keep_values: bool,
        nz: int,
    ) -> OnePointResult:
        if len(self.devices) > 1:
            require_host_arrays(f"OnePoint.{statistic}", inputs)
        edges_r = self._resolve_edges(edges, nz)
        per_scale: List[Dict[str, Any]] = []
        for s in range(self.nscale):
            parts = self._parts(kind, s)
            e_s = None if edges_r is None else edges_r[:, s]

            def fn(eng: Any, _i: int, e_s: Optional[np.ndarray] = e_s) -> Any:
                vals = measure(eng)
                valid = _nonempty_apertures(eng, kind, w)
                return _reduce_scale(vals, valid, names, e_s, keep_values)

            partials = run_on_devices(parts, fn)
            total = partials[0]
            for p in partials[1:]:
                total = _add_scale(total, p)
            per_scale.append(total)
        return self._assemble(statistic, names, per_scale, edges_r, keep_values, nz)

    def _assemble(
        self,
        statistic: str,
        names: Tuple[str, ...],
        per_scale: List[Dict[str, Any]],
        edges: Optional[np.ndarray],
        keep_values: bool,
        nz: int,
    ) -> OnePointResult:
        def stack(key: str, name: str, axis: int) -> np.ndarray:
            return np.stack([p["fields"][name][key] for p in per_scale], axis=axis)

        count = {
            name: np.array(
                [p["fields"][name]["count"] for p in per_scale], dtype=np.float64
            )
            for name in names
        }
        power = {name: stack("power", name, 1) for name in names}
        cross = {
            name: {
                k: np.stack([p["fields"][name]["cross"][k] for p in per_scale], axis=1)
                for k in ("ij", "iij", "ijk")
            }
            for name in names
        }
        hist = under = over = None
        if edges is not None:
            hist = {name: stack("hist", name, 1) for name in names}
            under = {name: stack("under", name, 1) for name in names}
            over = {name: stack("over", name, 1) for name in names}
        values = None
        if keep_values:
            values = {
                name: [p["fields"][name]["values"] for p in per_scale] for name in names
            }
        return OnePointResult(
            statistic,
            self.scales.copy(),
            tuple(names),
            np.array([p["n"] for p in per_scale], dtype=np.int64),
            count,
            power,
            cross,
            onepoint_cross_index(nz),
            edges,
            hist,
            under,
            over,
            values,
        )

    def g(
        self,
        shear_maps: Any,
        w: Any,
        *,
        flip_g1: bool = False,
        flip_g2: bool = False,
        cross: bool = True,
        edges: Optional[Any] = None,
        values: bool = False,
    ) -> OnePointResult:
        """Aperture-mass statistics of ``shear_maps (nz, 2, npix | n_active)``
        with weights ``w (nz, npix | n_active)``: fields ``M_ap`` and, with
        ``cross=True``, ``M_x`` (the B-mode / systematics check; one more
        pass over the disc geometry).  ``flip_g1``/``flip_g2`` as for
        ``Integrated3PCF.gg``.  ``edges`` overrides the constructor's;
        ``values=True`` keeps the per-aperture values."""
        names: Tuple[str, ...] = ("M_ap", "M_x") if cross else ("M_ap",)
        aperture_filter = self.aperture_filter

        def measure(eng: _ApertureEngine) -> Dict[str, Any]:
            out = {
                "M_ap": eng.m_a(
                    shear_maps,
                    w,
                    aperture_filter=aperture_filter,
                    flip_g1=flip_g1,
                    flip_g2=flip_g2,
                    return_device=True,
                )
            }
            if cross:
                # gamma_x = gamma_1 sin 2phi - gamma_2 cos 2phi is the tangential
                # kernel applied to (gamma_2, -gamma_1): swap the components
                # and flip the second (the sign fold is free and exact).
                xp = np if isinstance(shear_maps, np.ndarray) else eng.backend.module
                swapped = xp.ascontiguousarray(shear_maps[:, ::-1])
                out["M_x"] = eng.m_a(
                    swapped,
                    w,
                    aperture_filter=aperture_filter,
                    flip_g1=flip_g2,
                    flip_g2=not flip_g1,
                    return_device=True,
                )
            return out

        return self._measure(
            "g", "shear", names, measure, (shear_maps, w), w, edges, values, self._nz(w)
        )

    def k(
        self,
        maps: Any,
        w: Any,
        *,
        edges: Optional[Any] = None,
        values: bool = False,
    ) -> OnePointResult:
        """Statistics of the filtered scalar field (field ``k``) of ``maps
        (nz, npix | n_active)`` -- κ or δ -- with weights ``w``: the circular
        top hat, the HEALPix-pixel top hat or the filter U chosen by
        ``k_aperture_filter``."""
        kind = self._k_kind
        filt: Optional[Callable[..., Any]] = (
            self.k_aperture_filter if callable(self.k_aperture_filter) else None
        )

        def measure(eng: _ApertureEngine) -> Dict[str, Any]:
            return {"k": eng.m_g(maps, w, aperture_filter=filt, return_device=True)}

        return self._measure(
            "k", kind, ("k",), measure, (maps, w), w, edges, values, self._nz(w)
        )

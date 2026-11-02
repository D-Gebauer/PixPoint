"""
Labelled results of the three estimators.

Every measurement returns an object whose fields are named after what they
hold, together with the row labels (which tomographic bins) and the angular
binning -- instead of a positional tuple whose order and row layout the
caller has to know.

* :class:`IntegratedResult` -- per-patch outputs of
  :class:`~pixpoint.correlations.Integrated3PCF` (the i3PCF ingredients);
* :class:`TwoPointResult` -- summed numerators and weights of
  :class:`~pixpoint.global_2pcf.Global2PCF`;
* :class:`ThreePointResult` -- multipole sums of
  :class:`~pixpoint.global_3pcf.Global3PCF`;
* :class:`OnePointResult` -- histogram and moment sums of
  :class:`~pixpoint.onepoint.OnePoint`.

The global and one-point results hold *sums* (numerators and weights), so results of
different devices, footprints or chunks add with ``+``; the estimate is formed
on access.  Array fields may live on the host (numpy) or the device (cupy),
following the measurement's ``return_device``.
"""

import itertools
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backend import safe_divide
from .correlation_helpers import _xp

Combinations = Tuple[Tuple[int, int], ...]

# Field order of IntegratedResult (and of ZetaWriter datasets): the central
# aperture fields, then the annular 2PCFs.
INTEGRATED_FIELDS: Tuple[str, ...] = ("M_a", "M_g", "xi_p", "xi_m", "xi_g", "xi_t")


def _theta_arcmin(binedges: np.ndarray) -> np.ndarray:
    e = np.degrees(np.asarray(binedges, dtype=np.float64)) * 60.0
    return np.sqrt(e[1:] * e[:-1])


# ── Integrated3PCF ────────────────────────────────────────────────────────


@dataclass
class IntegratedResult:
    """Per-patch aperture masses and 2PCFs (the i3PCF ingredients).

    Fields that were not measured are ``None``.

    * ``M_a`` -- aperture mass of the shear, ``(nz_source, n_patches)``;
    * ``M_g`` -- aperture of the density, ``(nz_lens, n_patches)``;
    * ``xi_p``, ``xi_m`` -- ξ± per patch, ``(ncomb, n_patches, nbins)``, rows
      labelled by ``gg_combinations``;
    * ``xi_g`` -- w(θ) per patch, rows ``kk_combinations``;
    * ``xi_t`` -- γ_t per patch, rows ``kg_combinations`` = (lens, source).
    """

    binedges: np.ndarray
    M_a: Optional[Any] = None
    M_g: Optional[Any] = None
    xi_p: Optional[Any] = None
    xi_m: Optional[Any] = None
    xi_g: Optional[Any] = None
    xi_t: Optional[Any] = None
    gg_combinations: Optional[Combinations] = None
    kk_combinations: Optional[Combinations] = None
    kg_combinations: Optional[Combinations] = None

    @property
    def theta(self) -> np.ndarray:
        """Geometric bin centres (arcmin)."""
        return _theta_arcmin(self.binedges)

    def fields(self) -> Dict[str, Any]:
        """The measured arrays, by name, in the order of ``INTEGRATED_FIELDS``."""
        return {
            k: getattr(self, k)
            for k in INTEGRATED_FIELDS
            if getattr(self, k) is not None
        }

    def zetas(self, **kwargs: Any) -> Dict[str, Any]:
        """Every i3PCF ζ the measured fields form (see
        :func:`~pixpoint.correlation_helpers.calculate_all_zetas`; extra
        keyword arguments are forwarded)."""
        from .correlation_helpers import calculate_all_zetas

        batched = {k: v[None, ...] for k, v in self.fields().items()}
        return {k: v[0] for k, v in calculate_all_zetas(**batched, **kwargs).items()}

    def to_numpy(self) -> "IntegratedResult":
        """A copy with every array on the host."""
        out = {}
        for f in dc_fields(self):
            v = getattr(self, f.name)
            if f.name in INTEGRATED_FIELDS and v is not None:
                v = (
                    v.get()
                    if hasattr(v, "get") and not isinstance(v, np.ndarray)
                    else np.asarray(v)
                )
            out[f.name] = v
        return IntegratedResult(**out)


# ── Global2PCF ────────────────────────────────────────────────────────────


@dataclass
class TwoPointResult:
    """A footprint-wide 2PCF as sums over every pixel pair.

    ``statistic``: ``"gg"`` (fields ``xi_p``, ``xi_m``), ``"kk"`` (``xi``, =
    w(θ)) or ``"kg"`` (``xi``, = γ_t).  ``num[name]`` and ``weight``:
    ``(ncomb, nbins)`` float64 sums of ``w_i w_j f_i f_j`` and of ``w_i w_j``
    over the pairs of every bin; rows labelled by ``combinations``.  The
    estimate is their ratio (``estimate``, or the field properties).  Results
    of different devices or footprints add with ``+``.
    """

    statistic: str
    binedges: np.ndarray
    combinations: Combinations
    num: Dict[str, Any]
    weight: Any

    @property
    def theta(self) -> np.ndarray:
        """Geometric bin centres (arcmin)."""
        return _theta_arcmin(self.binedges)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.num)

    def estimate(self, name: str) -> Any:
        """``num[name] / weight`` (0 where no pairs)."""
        xp = _xp(self.weight)
        return safe_divide(xp, self.num[name], self.weight)

    @property
    def xi_p(self) -> Any:
        return self._field("xi_p")

    @property
    def xi_m(self) -> Any:
        return self._field("xi_m")

    @property
    def xi(self) -> Any:
        return self._field("xi")

    def _field(self, name: str) -> Any:
        if name not in self.num:
            raise AttributeError(
                f"a {self.statistic!r} result holds {self.names}, not {name!r}"
            )
        return self.estimate(name)

    def to_numpy(self) -> "TwoPointResult":
        def host(a: Any) -> np.ndarray:
            return (
                a.get()
                if hasattr(a, "get") and not isinstance(a, np.ndarray)
                else np.asarray(a)
            )

        return TwoPointResult(
            self.statistic,
            self.binedges,
            self.combinations,
            {k: host(v) for k, v in self.num.items()},
            host(self.weight),
        )

    def __add__(self, other: "TwoPointResult") -> "TwoPointResult":
        if (
            other.statistic != self.statistic
            or other.combinations != self.combinations
            or not np.array_equal(other.binedges, self.binedges)
        ):
            raise ValueError("can only add results of the same statistic and binning")
        return TwoPointResult(
            self.statistic,
            self.binedges,
            self.combinations,
            {k: v + other.num[k] for k, v in self.num.items()},
            self.weight + other.weight,
        )


# ── Global3PCF ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Component:
    """One three-point function: which field sits at which vertex, and which
    vertices are complex conjugated.  ``fields = (centre, leg 2, leg 3)``."""

    label: str
    fields: Tuple[str, str, str]
    conj: Tuple[bool, bool, bool] = (False, False, False)


@dataclass
class ThreePointResult:
    """Multipoles of one or more three-point functions.

    ``num[label]`` and ``norm[label]``: complex arrays of shape
    ``(2 n_max + 1, nz_centre, nz_leg2, nz_leg3, nbins, nbins)`` -- axis 0 is
    ``n = -n_max .. n_max``, the last two are the radial bins of legs 2 and
    3.  Sums, not ratios: results of different tiles, chunks or devices add
    (``+``), and the estimate is formed in opening-angle bins
    (:meth:`binned`).
    """

    n_max: int
    binedges: np.ndarray
    components: Tuple[Component, ...]
    spins: Dict[str, int]
    num: Dict[str, np.ndarray]
    norm: Dict[str, np.ndarray]

    @property
    def n(self) -> np.ndarray:
        return np.arange(-self.n_max, self.n_max + 1)

    @property
    def theta(self) -> np.ndarray:
        """Geometric radial bin centres (arcmin)."""
        return _theta_arcmin(self.binedges)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(c.label for c in self.components)

    def __add__(self, other: "ThreePointResult") -> "ThreePointResult":
        if (
            other.n_max != self.n_max
            or other.components != self.components
            or not np.array_equal(other.binedges, self.binedges)
        ):
            raise ValueError("can only add multipoles of the same definition")
        return ThreePointResult(
            self.n_max,
            self.binedges,
            self.components,
            dict(self.spins),
            {k: self.num[k] + other.num[k] for k in self.num},
            {k: self.norm[k] + other.norm[k] for k in self.norm},
        )

    @staticmethod
    def _phi_coefficients(n: np.ndarray, phi_edges: np.ndarray) -> np.ndarray:
        """``c_n^k = (1/2pi) int_{bin k} exp(-i n phi) dphi``, shape (nphi, n)."""
        a = np.asarray(phi_edges[:-1], dtype=np.float64)[:, None]
        b = np.asarray(phi_edges[1:], dtype=np.float64)[:, None]
        nn = n[None, :].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            c: np.ndarray = (np.exp(-1j * nn * a) - np.exp(-1j * nn * b)) / (
                2j * np.pi * nn
            )
        c[:, n == 0] = (b - a) / (2.0 * np.pi)
        return c

    def binned(
        self, phi_edges: Optional[np.ndarray] = None, nphi: int = 16
    ) -> Dict[str, np.ndarray]:
        """Estimate in opening-angle bins (x-projection).

        ``phi_edges`` default: ``nphi`` equal bins on ``[-pi, pi]``.  Returns
        ``label -> (nz1, nz2, nz3, nbins, nbins, nphi)``; bins without
        triangles are 0.
        """
        if phi_edges is None:
            phi_edges = np.linspace(-np.pi, np.pi, nphi + 1)
        c = self._phi_coefficients(self.n, np.asarray(phi_edges))
        out = {}
        for comp in self.components:
            # sum_n c_n Gamma_{-n}: reverse the n axis
            num = np.einsum("kn,n...->...k", c, self.num[comp.label][::-1])
            den = np.einsum("kn,n...->...k", c, self.norm[comp.label][::-1])
            with np.errstate(divide="ignore", invalid="ignore"):
                val = np.where(den != 0, num / den, 0.0)
            out[comp.label] = val
        return out


# ── OnePoint ──────────────────────────────────────────────────────────────

# Raw-moment orders the one-point reduction accumulates: Σ x^k, k = 1..5.
ONEPOINT_POWERS = 5


def onepoint_cross_index(nz: int) -> Dict[str, Tuple[Tuple[int, ...], ...]]:
    """Tomographic index tuples of the cross moments: ``"ij"`` ⟨x_i x_j⟩ for
    i < j, ``"iij"`` ⟨x_i² x_j⟩ for i ≠ j, ``"ijk"`` ⟨x_i x_j x_k⟩ for
    i < j < k (the moment set of the DES Y3 moments analysis)."""
    z = range(int(nz))
    return {
        "ij": tuple(itertools.combinations(z, 2)),
        "iij": tuple((i, j) for i in z for j in z if i != j),
        "ijk": tuple(itertools.combinations(z, 3)),
    }


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` broadcast, 0 where ``den`` is 0."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(num.shape, den.shape), dtype=np.float64)
    den_b = np.broadcast_to(den, out.shape)
    np.divide(np.broadcast_to(num, out.shape), den_b, out=out, where=den_b != 0)
    return out


@dataclass
class OnePointResult:
    """One-point statistics of aperture fields, as *sums over apertures*.

    ``statistic``: ``"g"`` (fields ``M_ap`` and ``M_x``, the tangential and
    cross aperture mass) or ``"k"`` (field ``k``, the filtered scalar field).
    Every array is host float64 and a sum over the accepted apertures of
    each scale, so results of different devices or footprints add with
    ``+``; the estimates are formed on access.

    * ``scales`` -- the aperture scales θ (arcmin), ``(nscale,)``;
    * ``n_apertures`` -- accepted centres per scale, ``(nscale,)``;
    * ``count[name]`` -- apertures that entered the sums (weight ``Σ w > 0``
      and a finite value in every tomographic bin), ``(nscale,)``;
    * ``power_sums[name]`` -- Σ x^k, k = 1..5, ``(nz, nscale, 5)``;
    * ``cross_sums[name][key]`` -- Σ x_i x_j (``"ij"``), Σ x_i² x_j
      (``"iij"``), Σ x_i x_j x_k (``"ijk"``) over the tomographic tuples
      ``cross_index[key]``, ``(ncomb, nscale)``;
    * ``edges`` -- histogram edges ``(nz, nscale, nbins + 1)``; ``hist[name]``
      ``(nz, nscale, nbins)`` counts on the half-open bins ``[e_b, e_b+1)``,
      ``underflow[name]`` / ``overflow[name]`` ``(nz, nscale)`` the values
      below the first / at or above the last edge (all ``None`` without
      edges);
    * ``values[name]`` -- the raw per-aperture values, one ``(nz,
      n_apertures)`` array per scale, when asked for (``values=True``).
    """

    statistic: str
    scales: np.ndarray
    names: Tuple[str, ...]
    n_apertures: np.ndarray
    count: Dict[str, np.ndarray]
    power_sums: Dict[str, np.ndarray]
    cross_sums: Dict[str, Dict[str, np.ndarray]]
    cross_index: Dict[str, Tuple[Tuple[int, ...], ...]]
    edges: Optional[np.ndarray] = None
    hist: Optional[Dict[str, np.ndarray]] = None
    underflow: Optional[Dict[str, np.ndarray]] = None
    overflow: Optional[Dict[str, np.ndarray]] = None
    values: Optional[Dict[str, List[np.ndarray]]] = None

    @property
    def nz(self) -> int:
        return int(next(iter(self.power_sums.values())).shape[0])

    @property
    def nscale(self) -> int:
        return int(np.asarray(self.scales).size)

    def moments(self, name: str) -> np.ndarray:
        """Raw moments ⟨x^k⟩, k = 1..5: ``(nz, nscale, 5)`` (0 without apertures)."""
        return _ratio(self.power_sums[name], self.count[name][None, :, None])

    def central_moments(self, name: str) -> np.ndarray:
        """Central moments ⟨(x − ⟨x⟩)^k⟩, k = 2..5: ``(nz, nscale, 4)``."""
        m = self.moments(name)
        m1, m2, m3, m4, m5 = (m[..., k] for k in range(5))
        c2 = m2 - m1**2
        c3 = m3 - 3 * m1 * m2 + 2 * m1**3
        c4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
        c5 = m5 - 5 * m1 * m4 + 10 * m1**2 * m3 - 10 * m1**3 * m2 + 4 * m1**5
        return np.stack([c2, c3, c4, c5], axis=-1)

    def cross_moments(self, name: str) -> Dict[str, np.ndarray]:
        """``{"ij": ⟨x_i x_j⟩, "iij": ⟨x_i² x_j⟩, "ijk": ⟨x_i x_j x_k⟩}``, each
        ``(ncomb, nscale)`` over ``cross_index``."""
        return {
            k: _ratio(v, self.count[name][None, :])
            for k, v in self.cross_sums[name].items()
        }

    def pdf(self, name: str) -> np.ndarray:
        """``hist / (count · bin width)``: ``(nz, nscale, nbins)``."""
        if self.hist is None or self.edges is None:
            raise ValueError("no histogram was measured: pass edges")
        widths = np.diff(self.edges, axis=-1)
        return _ratio(self.hist[name], self.count[name][None, :, None] * widths)

    def __add__(self, other: "OnePointResult") -> "OnePointResult":
        same_edges = (self.edges is None and other.edges is None) or (
            self.edges is not None
            and other.edges is not None
            and np.array_equal(self.edges, other.edges)
        )
        if (
            other.statistic != self.statistic
            or other.names != self.names
            or not np.array_equal(other.scales, self.scales)
            or not same_edges
        ):
            raise ValueError(
                "can only add one-point results of the same statistic, scales "
                "and histogram edges"
            )

        def add(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Any:
            if a is None or b is None:
                return None
            return {k: a[k] + b[k] for k in a}

        values = None
        if self.values is not None and other.values is not None:
            values = {
                k: [
                    np.concatenate((x, y), axis=1)
                    for x, y in zip(self.values[k], other.values[k])
                ]
                for k in self.values
            }
        return OnePointResult(
            self.statistic,
            self.scales,
            self.names,
            self.n_apertures + other.n_apertures,
            add(self.count, other.count),
            add(self.power_sums, other.power_sums),
            {k: add(self.cross_sums[k], other.cross_sums[k]) for k in self.cross_sums},
            dict(self.cross_index),
            self.edges,
            add(self.hist, other.hist),
            add(self.underflow, other.underflow),
            add(self.overflow, other.overflow),
            values,
        )

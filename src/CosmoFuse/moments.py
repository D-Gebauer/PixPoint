"""
Moment-carrying treecode cells: geometry and the reference arithmetic.

The static treecode replaces every child pair ``(i in I, j in J)`` of a
coarse cell pair by the parents' geometry.  For xi- the only approximation,
given the parents' bin, is the rotation phase: with the CosmoFuse rotation
``gamma' = gamma e^{2i phi}`` the pair product is

    gamma'_i gamma'_j = gamma_i gamma_j e^{2i(phi_I + phi_J)} e^{2i dphi_ij},

where ``dphi_ij = (phi_i + phi_j) - (phi_I + phi_J)`` is the change of the
pair's position angles for the children's offsets from the cell centroids
(to first order both ends turn by the same angle, hence the factor 2 on top
of the spin-2 factor 2 -- ``4 dphi`` for xi-, ``2 dphi`` for gamma_t, where
only the source end carries a phase).  Expanding ``e^{2i dphi}`` in the
offsets, the cell pair becomes a contraction of per-cell *moments* of the
weighted shear,

    M = sum_j w_j g~_j,   D = sum_j w_j g~_j dr_j,   Q = sum_j w_j g~_j dr_j dr_j^T,

with ``dr_j`` the gnomonic (tangent-plane) offset of child j from the cell
centroid in the cell's east/north frame and ``g~_j = g_j e^{2iS rho_j}`` the
child shear parallel-transported into that frame (``rho_j`` = angle of the
child's east vector in the cell frame; exact geometry, independent of the
maps).  ``S`` is the sign of the rotation convention: CosmoFuse rotates
with ``e^{+2i phi}``, which is ``S = -1`` in the notation of the prototype
(``benchmarks/moments/moment_probe.py``, S = +1 there with the opposite
rotation); every formula below carries the symbol so the derivation can be
read off.

Per oriented cell pair ``I -> J`` with separation ``d``, the unit vectors
perpendicular (p) and parallel (q) to the connecting geodesic at the I end
are ``p_I = (-sin phi_I, cos phi_I)``, ``q_I = (cos phi_I, sin phi_I)``; at
the J end they are the ones *transported along the geodesic*, ``p_J = (sin
phi_J, -cos phi_J)``, ``q_J = (-cos phi_J, -sin phi_J)``.  With ``Dp = p.D``
etc. the second-order factor is

    F = M_I M_J
        - 4iS/d   (M_I Dp_J - Dp_I M_J)
        - 8/d^2   (M_I Qpp_J - 2 Dp_I Dp_J + Qpp_I M_J)
        + 4iS/d^2 (M_I Qqp_J - Dq_J Dp_I - Dq_I Dp_J + Qqp_I M_J)

and ``xi- += Re(F e^{-2iS(phi_I + phi_J)})``, while xi+ needs only the
transported monopoles.  gamma_t has the same window with half the phase
(``4 -> 2``) and the *lens* cell contributing its scalar moments ``D_delta =
sum w delta dr``, ``Q_delta = sum w delta dr dr^T`` (no transport).

Everything here is host-side numpy at float64 and is the reference the
kernels (CUDA, Numba, the emulation twins) are gated against; the device
side normalises the moments by the cell weight, ``d = D/W``, ``q = Q/W``,
so that the pair kernels keep multiplying by ``w_a w_b`` as they do for the
weighted means.
"""

from typing import Any, Tuple

import numpy as np

#: Sign of the rotation convention in the expansion above: CosmoFuse rotates
#: the shear into the pair frame with ``e^{+2i phi}`` (``S = -1``).  Not a
#: user option -- see ``tests/test_cell_moments.py`` for the gate that fixes it.
MOMENT_SIGN = -1.0

#: Moments per cell for ``cell_moments="dipole"`` / ``"quad"``: M, D (2), Q (3).
N_MOMENTS = {1: 3, 2: 6}

MOMENT_ORDERS = {None: 0, "dipole": 1, "quad": 2}


def validate_cell_moments(cell_moments: Any) -> Any:
    """``None``, ``"dipole"`` or ``"quad"`` (``False`` -> ``None``)."""
    if cell_moments is None or cell_moments is False:
        return None
    if isinstance(cell_moments, str) and cell_moments.lower() in ("dipole", "quad"):
        return cell_moments.lower()
    raise ValueError(
        "cell_moments must be None (plain treecode cells), 'dipole' or 'quad'; "
        f"got {cell_moments!r}"
    )


# ── frames and offsets ───────────────────────────────────────────────────


def unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """``(n, 3)`` unit vectors of ``(ra, dec)`` (radians)."""
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    cd = np.cos(dec)
    return np.stack((cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)), axis=-1)


def frames(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Local east and north unit vectors of unit vectors ``xyz`` ``(n, 3)``.

    Without trigonometry: ``e = (-y, x, 0) / cos dec`` and ``n = (-z x, -z y,
    cos^2 dec) / cos dec`` with ``cos dec = sqrt(x^2 + y^2)`` -- the same
    arithmetic the device kernels use on the stored unit vectors.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    cd = np.sqrt(x * x + y * y)
    inv = np.where(cd > 0, 1.0 / np.where(cd > 0, cd, 1.0), 0.0)
    e = np.stack((-y * inv, x * inv, np.zeros_like(x)), axis=-1)
    n = np.stack((-z * x * inv, -z * y * inv, cd), axis=-1)
    # at a pole (cd == 0) the frame is undefined; fall back to the x axis
    pole = cd == 0
    if np.any(pole):
        e[pole] = (1.0, 0.0, 0.0)
        n[pole] = (0.0, 1.0, 0.0)
    return e, n


def cell_frames(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """``(n, 9)``: unit vector, east and north of every cell, concatenated --
    the per-cell geometry record the moment kernels read."""
    xyz = unit_vectors(ra, dec)
    e, n = frames(xyz)
    return np.ascontiguousarray(np.concatenate((xyz, e, n), axis=-1))


def moment_factors(
    cell_geom: np.ndarray, pix_xyz: np.ndarray, sign: float = MOMENT_SIGN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transport phasor and gnomonic offsets of children in their cell frame.

    Args:
        cell_geom: ``(nnz, 9)`` cell record (``cell_frames``) of every child's
            cell, i.e. already gathered per CSR entry.
        pix_xyz: ``(nnz, 3)`` unit vectors of the children.
        sign: ``S`` of the module docstring.

    Returns ``(t, dx, dy)``: ``t = e^{2iS rho}`` (complex128), ``dx``, ``dy``
    in radians (float64).
    """
    cell_geom = np.asarray(cell_geom, dtype=np.float64)
    pix_xyz = np.asarray(pix_xyz, dtype=np.float64)
    xc, ec, nc = cell_geom[:, 0:3], cell_geom[:, 3:6], cell_geom[:, 6:9]
    dot = np.einsum("ij,ij->i", xc, pix_xyz)
    inv = 1.0 / dot
    dx = np.einsum("ij,ij->i", ec, pix_xyz) * inv
    dy = np.einsum("ij,ij->i", nc, pix_xyz) * inv
    ep, _ = frames(pix_xyz)
    rho = np.arctan2(np.einsum("ij,ij->i", nc, ep), np.einsum("ij,ij->i", ec, ep))
    t = np.exp(2j * sign * rho)
    return t, dx, dy


def pair_angles(
    geom_a: np.ndarray, geom_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full position angles at both ends and the inverse separation.

    ``geom_a``, ``geom_b``: ``(n, 9)`` cell records (``cell_frames``) of the
    two ends.  ``phi_a`` is the position angle of b seen from a, from local
    east towards local north (the frame of the shear components, as in the
    pair finders), ``phi_b`` the one of a seen from b.
    """
    geom_a = np.asarray(geom_a, dtype=np.float64)
    geom_b = np.asarray(geom_b, dtype=np.float64)
    xa, ea, na = geom_a[:, 0:3], geom_a[:, 3:6], geom_a[:, 6:9]
    xb, eb, nb = geom_b[:, 0:3], geom_b[:, 3:6], geom_b[:, 6:9]
    phi_a = np.arctan2(np.einsum("ij,ij->i", na, xb), np.einsum("ij,ij->i", ea, xb))
    phi_b = np.arctan2(np.einsum("ij,ij->i", nb, xa), np.einsum("ij,ij->i", eb, xa))
    cross = np.cross(xa, xb)
    d = np.arctan2(
        np.sqrt(np.einsum("ij,ij->i", cross, cross)), np.einsum("ij,ij->i", xa, xb)
    )
    with np.errstate(divide="ignore"):
        inv_d = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    return phi_a, phi_b, inv_d


# ── reference moments and pair sums (numpy, float64) ─────────────────────


def cell_moments_numpy(
    indptr: np.ndarray,
    indices: np.ndarray,
    t: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    weights: np.ndarray,
    values: Any,
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unnormalised moments ``(n_cells, NM)`` of ``w * v * t`` and the weight
    sums ``(n_cells,)`` for one level, from the direct pixel CSR.

    ``values`` is complex (shear, transported with ``t``) or real (scalar,
    ``t`` is ignored) or ``None`` (weights only: moments of ``w``).  The
    moment order is ``[1, dx, dy, dx^2, dx dy, dy^2]`` truncated to
    ``N_MOMENTS[order]``.
    """
    n_cells = int(indptr.size - 1)
    nm = N_MOMENTS[order]
    cell = np.repeat(np.arange(n_cells), np.diff(indptr))
    w = np.asarray(weights, dtype=np.float64)[indices]
    if values is None:
        f = w.astype(np.complex128)
    else:
        v = np.asarray(values)[indices]
        if np.iscomplexobj(v):
            f = w * v.astype(np.complex128) * t
        else:
            f = (w * v.astype(np.float64)).astype(np.complex128)
    facs = [np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy][:nm]
    out = np.zeros((n_cells, nm), dtype=np.complex128)
    for m, fac in enumerate(facs):
        g = f * fac
        out[:, m] = np.bincount(cell, weights=np.real(g), minlength=n_cells) + 1j * (
            np.bincount(cell, weights=np.imag(g), minlength=n_cells)
        )
    W = np.bincount(cell, weights=w, minlength=n_cells)
    return out, W


def pair_factor(
    mom_i: np.ndarray,
    mom_j: np.ndarray,
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    inv_d: np.ndarray,
    order: int,
    spin_factor: float = 4.0,
    sign: float = MOMENT_SIGN,
) -> np.ndarray:
    """The correction factor ``F`` of the module docstring for oriented pairs
    ``I -> J`` (``spin_factor`` 4 for xi-, 2 for gamma_t).

    ``mom_i``, ``mom_j``: ``(n, NM)`` *unnormalised* or *normalised*
    moments of the two ends (the formula is bilinear, so either works as
    long as both ends agree); complex, scalar moments with zero imaginary
    part are fine.  The I end uses the plain unit vectors, the J end the
    transported ones.
    """
    phi_i = np.asarray(phi_i, dtype=np.float64)
    phi_j = np.asarray(phi_j, dtype=np.float64)
    si, ci, sj, cj = np.sin(phi_i), np.cos(phi_i), np.sin(phi_j), np.cos(phi_j)
    MI, MJ = mom_i[:, 0], mom_j[:, 0]
    F = MI * MJ
    if order >= 1:
        idd = np.asarray(inv_d, dtype=np.float64)
        pIx, pIy, qIx, qIy = -si, ci, ci, si
        pJx, pJy, qJx, qJy = sj, -cj, -cj, -sj
        DpI = mom_i[:, 1] * pIx + mom_i[:, 2] * pIy
        DqI = mom_i[:, 1] * qIx + mom_i[:, 2] * qIy
        DpJ = mom_j[:, 1] * pJx + mom_j[:, 2] * pJy
        DqJ = mom_j[:, 1] * qJx + mom_j[:, 2] * qJy
        c1 = spin_factor
        F = F + 1j * (MI * DpJ - DpI * MJ) * (-c1 * sign * idd)
        if order >= 2:
            QppI = (
                mom_i[:, 3] * pIx * pIx
                + mom_i[:, 4] * 2.0 * pIx * pIy
                + mom_i[:, 5] * pIy * pIy
            )
            QqpI = (
                mom_i[:, 3] * qIx * pIx
                + mom_i[:, 4] * (qIx * pIy + qIy * pIx)
                + mom_i[:, 5] * qIy * pIy
            )
            QppJ = (
                mom_j[:, 3] * pJx * pJx
                + mom_j[:, 4] * 2.0 * pJx * pJy
                + mom_j[:, 5] * pJy * pJy
            )
            QqpJ = (
                mom_j[:, 3] * qJx * pJx
                + mom_j[:, 4] * (qJx * pJy + qJy * pJx)
                + mom_j[:, 5] * qJy * pJy
            )
            c2 = 0.5 * c1 * c1  # (c1 dphi)^2 / 2
            F = F + (MI * QppJ - 2.0 * DpI * DpJ + QppI * MJ) * (-c2 * idd * idd)
            F = F + 1j * (MI * QqpJ - DqJ * DpI - DqI * DpJ + QqpI * MJ) * (
                c1 * sign * idd * idd
            )
    return np.asarray(F)


def xipm_pair_terms(
    mom_i: np.ndarray,
    mom_j: np.ndarray,
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    inv_d: np.ndarray,
    order: int,
    sign: float = MOMENT_SIGN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pair xi+ and xi- terms (without the weights) of oriented cell
    pairs ``I -> J`` from their moments."""
    F = pair_factor(mom_i, mom_j, phi_i, phi_j, inv_d, order, 4.0, sign)
    ri = np.exp(-2j * sign * np.asarray(phi_i, dtype=np.float64))
    rj = np.exp(-2j * sign * np.asarray(phi_j, dtype=np.float64))
    xim = np.real(F * ri * rj)
    xip = np.real(mom_i[:, 0] * np.conj(mom_j[:, 0]) * ri * np.conj(rj))
    return xip, xim


def gammat_pair_terms(
    mom_lens: np.ndarray,
    mom_src: np.ndarray,
    phi_lens: np.ndarray,
    phi_src: np.ndarray,
    inv_d: np.ndarray,
    order: int,
    sign: float = MOMENT_SIGN,
) -> np.ndarray:
    """Per-pair gamma_t terms (without the weights) of oriented cell pairs
    ``lens -> source``: ``-Re(F_t e^{-2iS phi_src})`` with the lens moments
    real and untransported."""
    F = pair_factor(mom_lens, mom_src, phi_lens, phi_src, inv_d, order, 2.0, sign)
    rj = np.exp(-2j * sign * np.asarray(phi_src, dtype=np.float64))
    return np.asarray(-np.real(F * rj))


# ── aperture statistics on a coarse aperture level ───────────────────────


def aperture_shear_terms(
    mom: np.ndarray,
    psi: np.ndarray,
    inv_theta: np.ndarray,
    q_val: np.ndarray,
    q_dval: np.ndarray,
    q_ddval: np.ndarray,
    order: int,
    sign: float = MOMENT_SIGN,
) -> np.ndarray:
    """Per disc cell, the numerator term of the aperture mass ``M_a`` from the
    cell's moments: ``sum_p w_p gamma_t(p) Q(theta_p)`` to ``order`` in the
    children's offsets.

    The aperture is the gamma_t pair statistic between the centre (a point,
    no moments) and the cell: with ``psi`` the position angle of the centre
    seen from the cell (east -> north, the pair-finder convention; the
    aperture geometry's bearing ``phi`` is ``pi - psi``, so ``e^{-2i phi} =
    e^{2i psi}``), ``d = theta`` the centre-cell distance and the transported
    unit vectors ``p = (sin psi, -cos psi)``, ``q = (-cos psi, -sin psi)``,

        gamma_t(p) Q(theta_p) = -Re[ gamma~_p e^{2i psi} e^{2iS dpsi_p} ] Q(theta_p),
        theta_p = theta + q.dr + (p.dr)^2 / (2 theta),
        Q(theta_p) = Q + Q' q.dr + Q'' (q.dr)^2 / 2 + (Q' / theta) (p.dr)^2 / 2,

    and ``dpsi`` the gamma_t phase shift with the centre end fixed.  Summed
    over the children with the moments of ``w gamma~`` (``M, Dp, Dq, Qpp,
    Qqp, Qqq``):

        N = -Re{ e^{2i psi} [ Q F_pt + Q' (Dq + 2iS/d Qqp) + Q'' Qqq / 2
                              + (Q' / d) Qpp / 2 ] },
        F_pt = M - 2iS/d Dp - 2/d^2 Qpp + 2iS/d^2 Qqp   (the lens a point).

    ``mom``: ``(n, NM)`` complex moments ``[M, dx, dy, (xx, xy, yy)]`` of
    ``w gamma~`` (or normalised by ``W``; the caller multiplies back).
    """
    psi = np.asarray(psi, dtype=np.float64)
    s, c = np.sin(psi), np.cos(psi)
    rot = np.exp(2j * psi)  # e^{-2iS psi}, S = -1
    M = mom[:, 0]
    N = q_val * M
    if order >= 1:
        idd = np.asarray(inv_theta, dtype=np.float64)
        px, py, qx, qy = s, -c, -c, -s
        Dp = mom[:, 1] * px + mom[:, 2] * py
        Dq = mom[:, 1] * qx + mom[:, 2] * qy
        c1 = 2.0
        N = N + q_val * (1j * Dp * (-c1 * sign * idd)) + q_dval * Dq
        if order >= 2:
            Qpp = mom[:, 3] * px * px + mom[:, 4] * 2.0 * px * py + mom[:, 5] * py * py
            Qqp = (
                mom[:, 3] * qx * px
                + mom[:, 4] * (qx * py + qy * px)
                + mom[:, 5] * qy * py
            )
            Qqq = mom[:, 3] * qx * qx + mom[:, 4] * 2.0 * qx * qy + mom[:, 5] * qy * qy
            N = N + q_val * (
                Qpp * (-0.5 * c1 * c1 * idd * idd) + 1j * Qqp * (c1 * sign * idd * idd)
            )
            N = N + q_dval * (1j * Qqp * (-c1 * sign * idd))
            N = N + 0.5 * q_ddval * Qqq + 0.5 * q_dval * idd * Qpp
    return np.asarray(-np.real(N * rot))


def aperture_scalar_terms(
    mom: np.ndarray,
    psi: np.ndarray,
    inv_theta: np.ndarray,
    u_val: np.ndarray,
    u_dval: np.ndarray,
    u_ddval: np.ndarray,
    order: int,
) -> np.ndarray:
    """Per disc cell, the numerator term of a scalar aperture ``M_g`` with a
    filter U from the cell's real moments of ``w f``: ``U M + U' Dq + U'' Qqq
    / 2 + (U' / theta) Qpp / 2`` (no phase).  ``psi`` as in
    :func:`aperture_shear_terms` (the radial direction is ``q``)."""
    psi = np.asarray(psi, dtype=np.float64)
    s, c = np.sin(psi), np.cos(psi)
    M = np.real(mom[:, 0])
    N = u_val * M
    if order >= 1:
        idd = np.asarray(inv_theta, dtype=np.float64)
        px, py, qx, qy = s, -c, -c, -s
        Dq = np.real(mom[:, 1]) * qx + np.real(mom[:, 2]) * qy
        N = N + u_dval * Dq
        if order >= 2:
            xx, xy, yy = (np.real(mom[:, k]) for k in (3, 4, 5))
            Qpp = xx * px * px + xy * 2.0 * px * py + yy * py * py
            Qqq = xx * qx * qx + xy * 2.0 * qx * qy + yy * qy * qy
            N = N + 0.5 * u_ddval * Qqq + 0.5 * u_dval * idd * Qpp
    return np.asarray(N)

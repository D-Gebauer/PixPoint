"""Moment-carrying treecode cells: the per-level moment builder and the
pair kernels of the coarse bins (``cell_moments="dipole" | "quad"``).

CPU: Numba kernels.  GPU: ``cuda/moment_rows.cu`` (the moment builder) and
the ``*_mom`` combination tiles of ``cuda/pair_tiles.cuh`` behind the
``gpu_tiled_{tomo,packed}_reduce_{xipm,ds,3x2pt}_mom`` kernels.

Contract shared by both backends (see ``pixpoint.moments`` for the
derivation and the sign convention):

* a cell row ``c = row - n_active`` carries, per tomographic bin, ``NM``
  normalised moments ``[M, dx, dy, (dx^2, dx dy, dy^2)] / W`` -- complex
  (interleaved re/im, transported) for shear, real for scalars -- in an AoS
  buffer ``(n_appended, n_lead, NM, NC)``; the row buffers hold the weight
  sum and, for shear with moments on, the *transported* mean ``M/W``;
* the coarse bins' pairs carry the full position angles at both ends and
  the inverse separation; the kernels rotate with ``e^{2i phi}`` built from
  ``phi`` (not from the stored ``e^{2i phi}``) so the packed and the
  unpacked geometry run the same arithmetic;
* the plain tiles run over every bin first (the coarse bins' plain sums are
  a small fraction of the work), then these kernels overwrite the coarse
  bins; ``mom_base[b] < 0`` marks a fine bin, which they skip.
"""

from typing import Any, Tuple

import numpy as np

from .._numba import njit, prange
from ..moments import MOMENT_SIGN
from .combinations import _MAX_TILED_ACCUMULATORS, _combination_layout
from .raw import KernelFile, _has_raw_cuda_compiler, c_index, c_real

_S = MOMENT_SIGN


# ── Numba: cell moments of one level ──────────────────────────────────────


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_moment_level_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    cell_geom: np.ndarray,
    pix_xyz: np.ndarray,
    w_src: np.ndarray,
    v_src: np.ndarray,
    kind: int,
    nm: int,
    out_mom: np.ndarray,
    out_w: np.ndarray,
) -> None:
    """Normalised moments of one level from the direct pixel CSR.

    ``w_src``: ``(n_lead, n_active)``; ``v_src``: ``(n_lead, 2, n_active)``
    for shear (``kind`` 2), ``(n_lead, 1, n_active)`` for a scalar (``kind``
    1), unused for weights only (``kind`` 0).  ``out_mom``: ``(n_cells,
    n_lead, nm, 2)`` at float64 -- the imaginary parts stay zero for real
    kinds; ``out_w``: ``(n_cells, n_lead)``.  Geometry at float64 from the
    stored unit vectors (``moments.moment_factors``).
    """
    n_cells = indptr.shape[0] - 1
    n_lead = w_src.shape[0]
    for c in prange(n_cells):
        xc0, xc1, xc2 = cell_geom[c, 0], cell_geom[c, 1], cell_geom[c, 2]
        ec0, ec1, ec2 = cell_geom[c, 3], cell_geom[c, 4], cell_geom[c, 5]
        nc0, nc1, nc2 = cell_geom[c, 6], cell_geom[c, 7], cell_geom[c, 8]
        for lead in range(n_lead):
            sw = 0.0
            acc_re = np.zeros(6)
            acc_im = np.zeros(6)
            for k in range(indptr[c], indptr[c + 1]):
                row = indices[k]
                px, py, pz = (
                    np.float64(pix_xyz[row, 0]),
                    np.float64(pix_xyz[row, 1]),
                    np.float64(pix_xyz[row, 2]),
                )
                dot = xc0 * px + xc1 * py + xc2 * pz
                inv = 1.0 / dot
                dx = (ec0 * px + ec1 * py + ec2 * pz) * inv
                dy = (nc0 * px + nc1 * py + nc2 * pz) * inv
                w = np.float64(w_src[lead, row])
                sw += w
                if kind == 2:
                    # frame of the child from its unit vector (no trig)
                    cd = np.sqrt(px * px + py * py)
                    if cd > 0.0:
                        ep0 = -py / cd
                        ep1 = px / cd
                    else:
                        ep0 = 1.0
                        ep1 = 0.0
                    rho = np.arctan2(nc0 * ep0 + nc1 * ep1, ec0 * ep0 + ec1 * ep1)
                    tr = np.cos(2.0 * _S * rho)
                    ti = np.sin(2.0 * _S * rho)
                    g1 = np.float64(v_src[lead, 0, row])
                    g2 = np.float64(v_src[lead, 1, row])
                    fr = w * (g1 * tr - g2 * ti)
                    fi = w * (g1 * ti + g2 * tr)
                elif kind == 1:
                    fr = w * np.float64(v_src[lead, 0, row])
                    fi = 0.0
                else:
                    fr = w
                    fi = 0.0
                acc_re[0] += fr
                acc_im[0] += fi
                acc_re[1] += fr * dx
                acc_im[1] += fi * dx
                acc_re[2] += fr * dy
                acc_im[2] += fi * dy
                if nm > 3:
                    acc_re[3] += fr * dx * dx
                    acc_im[3] += fi * dx * dx
                    acc_re[4] += fr * dx * dy
                    acc_im[4] += fi * dx * dy
                    acc_re[5] += fr * dy * dy
                    acc_im[5] += fi * dy * dy
            out_w[c, lead] = sw
            inv_w = 1.0 / sw if sw != 0.0 else 0.0
            for m in range(nm):
                out_mom[c, lead, m, 0] = acc_re[m] * inv_w
                out_mom[c, lead, m, 1] = acc_im[m] * inv_w


# ── Numba: pair kernels of the coarse bins ───────────────────────────────


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_xipm_tomo_moments_kernel(
    shear_map: np.ndarray,
    weights: np.ndarray,
    mom: np.ndarray,
    n_active: int,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    inv_d: np.ndarray,
    mom_base: np.ndarray,
    offsets: np.ndarray,
    comb_i: np.ndarray,
    comb_j: np.ndarray,
    order: int,
    out_p: np.ndarray,
    out_m: np.ndarray,
    out_w: np.ndarray,
) -> None:
    """xi+/xi- of the coarse bins with the moment correction (float64).

    ``shear_map`` ``(nrows, NS, 2)`` holds the transported cell means,
    ``mom`` ``(n_appended, NS, NM, 2)`` the normalised moments; pair ``idx``
    of bin ``b`` reads its angles at ``mom_base[b] + idx - offsets[b]``.
    Bins with ``mom_base[b] < 0`` (fine bins) are left untouched: the plain
    kernel owns them and has already written them.
    """
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]
    S = _S
    c1 = 4.0
    c2 = 8.0
    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]
        if mom_base[b] < 0 or stop <= start:
            continue  # a fine bin: the plain kernel owns it
        base = mom_base[b] - start
        acc_p = np.zeros(ncomb)
        acc_m = np.zeros(ncomb)
        acc_w = np.zeros(ncomb)
        for idx in range(start, stop):
            pix_i = int(ind_i[idx])
            pix_j = int(ind_j[idx])
            ci_ = pix_i - n_active
            cj_ = pix_j - n_active
            g = base + idx
            fi = np.float64(phi_i[g])
            fj = np.float64(phi_j[g])
            idd = np.float64(inv_d[g])
            si = np.sin(fi)
            ci = np.cos(fi)
            sj = np.sin(fj)
            cj = np.cos(fj)
            # e^{-2iS phi} at both ends (= e^{+2i phi} for S = -1)
            ri_r = ci * ci - si * si
            ri_i = -S * 2.0 * si * ci
            rj_r = cj * cj - sj * sj
            rj_i = -S * 2.0 * sj * cj
            # xi-: F e^{-2iS(phi_i + phi_j)}; xi+: M_i conj(M_j) e^{-2iS(phi_i - phi_j)}
            rij_r = ri_r * rj_r - ri_i * rj_i
            rij_i = ri_r * rj_i + ri_i * rj_r
            rijc_r = ri_r * rj_r + ri_i * rj_i
            rijc_i = ri_i * rj_r - ri_r * rj_i
            pIx, pIy, qIx, qIy = -si, ci, ci, si
            pJx, pJy, qJx, qJy = sj, -cj, -cj, -sj
            for comb_idx in range(ncomb):
                zi = comb_i[comb_idx]
                zj = comb_j[comb_idx]
                n_ori = 1 if zi == zj else 2
                for ori in range(n_ori):
                    za = zi if ori == 0 else zj
                    zb = zj if ori == 0 else zi
                    MI_r = np.float64(shear_map[pix_i, za, 0])
                    MI_i = np.float64(shear_map[pix_i, za, 1])
                    MJ_r = np.float64(shear_map[pix_j, zb, 0])
                    MJ_i = np.float64(shear_map[pix_j, zb, 1])
                    F_r = MI_r * MJ_r - MI_i * MJ_i
                    F_i = MI_r * MJ_i + MI_i * MJ_r
                    if order >= 1:
                        dxI_r = np.float64(mom[ci_, za, 1, 0])
                        dxI_i = np.float64(mom[ci_, za, 1, 1])
                        dyI_r = np.float64(mom[ci_, za, 2, 0])
                        dyI_i = np.float64(mom[ci_, za, 2, 1])
                        dxJ_r = np.float64(mom[cj_, zb, 1, 0])
                        dxJ_i = np.float64(mom[cj_, zb, 1, 1])
                        dyJ_r = np.float64(mom[cj_, zb, 2, 0])
                        dyJ_i = np.float64(mom[cj_, zb, 2, 1])
                        DpI_r = dxI_r * pIx + dyI_r * pIy
                        DpI_i = dxI_i * pIx + dyI_i * pIy
                        DqI_r = dxI_r * qIx + dyI_r * qIy
                        DqI_i = dxI_i * qIx + dyI_i * qIy
                        DpJ_r = dxJ_r * pJx + dyJ_r * pJy
                        DpJ_i = dxJ_i * pJx + dyJ_i * pJy
                        DqJ_r = dxJ_r * qJx + dyJ_r * qJy
                        DqJ_i = dxJ_i * qJx + dyJ_i * qJy
                        # A = M_I Dp_J - Dp_I M_J ;  F += i A (-c1 S / d)
                        A_r = (MI_r * DpJ_r - MI_i * DpJ_i) - (
                            DpI_r * MJ_r - DpI_i * MJ_i
                        )
                        A_i = (MI_r * DpJ_i + MI_i * DpJ_r) - (
                            DpI_r * MJ_i + DpI_i * MJ_r
                        )
                        f1 = -c1 * S * idd
                        F_r += -A_i * f1
                        F_i += A_r * f1
                        if order >= 2:
                            xxI_r = np.float64(mom[ci_, za, 3, 0])
                            xxI_i = np.float64(mom[ci_, za, 3, 1])
                            xyI_r = np.float64(mom[ci_, za, 4, 0])
                            xyI_i = np.float64(mom[ci_, za, 4, 1])
                            yyI_r = np.float64(mom[ci_, za, 5, 0])
                            yyI_i = np.float64(mom[ci_, za, 5, 1])
                            xxJ_r = np.float64(mom[cj_, zb, 3, 0])
                            xxJ_i = np.float64(mom[cj_, zb, 3, 1])
                            xyJ_r = np.float64(mom[cj_, zb, 4, 0])
                            xyJ_i = np.float64(mom[cj_, zb, 4, 1])
                            yyJ_r = np.float64(mom[cj_, zb, 5, 0])
                            yyJ_i = np.float64(mom[cj_, zb, 5, 1])
                            QppI_r = (
                                xxI_r * pIx * pIx
                                + xyI_r * 2.0 * pIx * pIy
                                + yyI_r * pIy * pIy
                            )
                            QppI_i = (
                                xxI_i * pIx * pIx
                                + xyI_i * 2.0 * pIx * pIy
                                + yyI_i * pIy * pIy
                            )
                            QqpI_r = (
                                xxI_r * qIx * pIx
                                + xyI_r * (qIx * pIy + qIy * pIx)
                                + yyI_r * qIy * pIy
                            )
                            QqpI_i = (
                                xxI_i * qIx * pIx
                                + xyI_i * (qIx * pIy + qIy * pIx)
                                + yyI_i * qIy * pIy
                            )
                            QppJ_r = (
                                xxJ_r * pJx * pJx
                                + xyJ_r * 2.0 * pJx * pJy
                                + yyJ_r * pJy * pJy
                            )
                            QppJ_i = (
                                xxJ_i * pJx * pJx
                                + xyJ_i * 2.0 * pJx * pJy
                                + yyJ_i * pJy * pJy
                            )
                            QqpJ_r = (
                                xxJ_r * qJx * pJx
                                + xyJ_r * (qJx * pJy + qJy * pJx)
                                + yyJ_r * qJy * pJy
                            )
                            QqpJ_i = (
                                xxJ_i * qJx * pJx
                                + xyJ_i * (qJx * pJy + qJy * pJx)
                                + yyJ_i * qJy * pJy
                            )
                            # B = M_I Qpp_J - 2 Dp_I Dp_J + Qpp_I M_J
                            # F += B (-c2 / d^2)
                            B_r = (
                                (MI_r * QppJ_r - MI_i * QppJ_i)
                                - 2.0 * (DpI_r * DpJ_r - DpI_i * DpJ_i)
                                + (QppI_r * MJ_r - QppI_i * MJ_i)
                            )
                            B_i = (
                                (MI_r * QppJ_i + MI_i * QppJ_r)
                                - 2.0 * (DpI_r * DpJ_i + DpI_i * DpJ_r)
                                + (QppI_r * MJ_i + QppI_i * MJ_r)
                            )
                            f2 = -c2 * idd * idd
                            F_r += B_r * f2
                            F_i += B_i * f2
                            # C = M_I Qqp_J - Dq_J Dp_I - Dq_I Dp_J + Qqp_I M_J
                            # F += i C (c1 S / d^2)
                            C_r = (
                                (MI_r * QqpJ_r - MI_i * QqpJ_i)
                                - (DqJ_r * DpI_r - DqJ_i * DpI_i)
                                - (DqI_r * DpJ_r - DqI_i * DpJ_i)
                                + (QqpI_r * MJ_r - QqpI_i * MJ_i)
                            )
                            C_i = (
                                (MI_r * QqpJ_i + MI_i * QqpJ_r)
                                - (DqJ_r * DpI_i + DqJ_i * DpI_r)
                                - (DqI_r * DpJ_i + DqI_i * DpJ_r)
                                + (QqpI_r * MJ_i + QqpI_i * MJ_r)
                            )
                            f3 = c1 * S * idd * idd
                            F_r += -C_i * f3
                            F_i += C_r * f3
                    w_pair = np.float64(weights[pix_i, za]) * np.float64(
                        weights[pix_j, zb]
                    )
                    xim = F_r * rij_r - F_i * rij_i
                    # G = M_I conj(M_J)
                    G_r = MI_r * MJ_r + MI_i * MJ_i
                    G_i = MI_i * MJ_r - MI_r * MJ_i
                    xip = G_r * rijc_r - G_i * rijc_i
                    acc_w[comb_idx] += w_pair
                    acc_p[comb_idx] += w_pair * xip
                    acc_m[comb_idx] += w_pair * xim
        for comb_idx in range(ncomb):
            out_p[comb_idx, b] = acc_p[comb_idx]
            out_m[comb_idx, b] = acc_m[comb_idx]
            out_w[comb_idx, b] = acc_w[comb_idx]


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_density_shear_tomo_moments_kernel(
    density_map: np.ndarray,
    shear_map: np.ndarray,
    lens_weights: np.ndarray,
    source_weights: np.ndarray,
    mom_lens: np.ndarray,
    mom_src: np.ndarray,
    n_active: int,
    ind_i: np.ndarray,
    ind_j: np.ndarray,
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    inv_d: np.ndarray,
    mom_base: np.ndarray,
    offsets: np.ndarray,
    comb_i: np.ndarray,
    comb_j: np.ndarray,
    order: int,
    out_num: np.ndarray,
    out_den: np.ndarray,
) -> None:
    """gamma_t of the coarse bins with the moment correction (float64).

    ``mom_lens`` ``(n_appended, ND, NM, 2)`` real scalar moments (imaginary
    parts zero), ``mom_src`` ``(n_appended, NS, NM, 2)`` the transported
    shear moments.  Each pair contributes in both orientations.
    """
    n_bins = offsets.shape[0] - 1
    ncomb = comb_i.shape[0]
    S = _S
    c1 = 2.0
    c2 = 2.0
    for b in prange(n_bins):
        start = offsets[b]
        stop = offsets[b + 1]
        if mom_base[b] < 0 or stop <= start:
            continue  # a fine bin: the plain kernel owns it
        base = mom_base[b] - start
        acc_n = np.zeros(ncomb)
        acc_w = np.zeros(ncomb)
        for idx in range(start, stop):
            pix_a = int(ind_i[idx])
            pix_b = int(ind_j[idx])
            g = base + idx
            fa = np.float64(phi_i[g])
            fb = np.float64(phi_j[g])
            idd = np.float64(inv_d[g])
            for ori in range(2):
                # ori 0: a = lens (I), b = source (J); ori 1: swapped
                if ori == 0:
                    pl, ps = pix_a, pix_b
                    fl, fs = fa, fb
                else:
                    pl, ps = pix_b, pix_a
                    fl, fs = fb, fa
                cl_ = pl - n_active
                cs_ = ps - n_active
                sl = np.sin(fl)
                cl = np.cos(fl)
                ss = np.sin(fs)
                cs = np.cos(fs)
                rs_r = cs * cs - ss * ss
                rs_i = -S * 2.0 * ss * cs
                pIx, pIy, qIx, qIy = -sl, cl, cl, sl
                pJx, pJy, qJx, qJy = ss, -cs, -cs, -ss
                for comb_idx in range(ncomb):
                    zl = comb_i[comb_idx]
                    zs = comb_j[comb_idx]
                    MI = np.float64(density_map[pl, zl])
                    MJ_r = np.float64(shear_map[ps, zs, 0])
                    MJ_i = np.float64(shear_map[ps, zs, 1])
                    F_r = MI * MJ_r
                    F_i = MI * MJ_i
                    if order >= 1:
                        dxI = np.float64(mom_lens[cl_, zl, 1, 0])
                        dyI = np.float64(mom_lens[cl_, zl, 2, 0])
                        dxJ_r = np.float64(mom_src[cs_, zs, 1, 0])
                        dxJ_i = np.float64(mom_src[cs_, zs, 1, 1])
                        dyJ_r = np.float64(mom_src[cs_, zs, 2, 0])
                        dyJ_i = np.float64(mom_src[cs_, zs, 2, 1])
                        DpI = dxI * pIx + dyI * pIy
                        DqI = dxI * qIx + dyI * qIy
                        DpJ_r = dxJ_r * pJx + dyJ_r * pJy
                        DpJ_i = dxJ_i * pJx + dyJ_i * pJy
                        DqJ_r = dxJ_r * qJx + dyJ_r * qJy
                        DqJ_i = dxJ_i * qJx + dyJ_i * qJy
                        A_r = MI * DpJ_r - DpI * MJ_r
                        A_i = MI * DpJ_i - DpI * MJ_i
                        f1 = -c1 * S * idd
                        F_r += -A_i * f1
                        F_i += A_r * f1
                        if order >= 2:
                            xxI = np.float64(mom_lens[cl_, zl, 3, 0])
                            xyI = np.float64(mom_lens[cl_, zl, 4, 0])
                            yyI = np.float64(mom_lens[cl_, zl, 5, 0])
                            xxJ_r = np.float64(mom_src[cs_, zs, 3, 0])
                            xxJ_i = np.float64(mom_src[cs_, zs, 3, 1])
                            xyJ_r = np.float64(mom_src[cs_, zs, 4, 0])
                            xyJ_i = np.float64(mom_src[cs_, zs, 4, 1])
                            yyJ_r = np.float64(mom_src[cs_, zs, 5, 0])
                            yyJ_i = np.float64(mom_src[cs_, zs, 5, 1])
                            QppI = (
                                xxI * pIx * pIx
                                + xyI * 2.0 * pIx * pIy
                                + yyI * pIy * pIy
                            )
                            QqpI = (
                                xxI * qIx * pIx
                                + xyI * (qIx * pIy + qIy * pIx)
                                + yyI * qIy * pIy
                            )
                            QppJ_r = (
                                xxJ_r * pJx * pJx
                                + xyJ_r * 2.0 * pJx * pJy
                                + yyJ_r * pJy * pJy
                            )
                            QppJ_i = (
                                xxJ_i * pJx * pJx
                                + xyJ_i * 2.0 * pJx * pJy
                                + yyJ_i * pJy * pJy
                            )
                            QqpJ_r = (
                                xxJ_r * qJx * pJx
                                + xyJ_r * (qJx * pJy + qJy * pJx)
                                + yyJ_r * qJy * pJy
                            )
                            QqpJ_i = (
                                xxJ_i * qJx * pJx
                                + xyJ_i * (qJx * pJy + qJy * pJx)
                                + yyJ_i * qJy * pJy
                            )
                            B_r = MI * QppJ_r - 2.0 * DpI * DpJ_r + QppI * MJ_r
                            B_i = MI * QppJ_i - 2.0 * DpI * DpJ_i + QppI * MJ_i
                            f2 = -c2 * idd * idd
                            F_r += B_r * f2
                            F_i += B_i * f2
                            C_r = MI * QqpJ_r - DqJ_r * DpI - DqI * DpJ_r + QqpI * MJ_r
                            C_i = MI * QqpJ_i - DqJ_i * DpI - DqI * DpJ_i + QqpI * MJ_i
                            f3 = c1 * S * idd * idd
                            F_r += -C_i * f3
                            F_i += C_r * f3
                    w_pair = np.float64(lens_weights[pl, zl]) * np.float64(
                        source_weights[ps, zs]
                    )
                    gt = -(F_r * rs_r - F_i * rs_i)
                    acc_w[comb_idx] += w_pair
                    acc_n[comb_idx] += w_pair * gt
        for comb_idx in range(ncomb):
            out_num[comb_idx, b] = acc_n[comb_idx]
            out_den[comb_idx, b] = acc_w[comb_idx]


# ── Numba: aperture statistics of moment-carrying aperture cells ──────────


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_aperture_shear_moments_kernel(
    Q_inds: np.ndarray,
    Q_phi: np.ndarray,
    Q_inv_theta: np.ndarray,
    Q_val: np.ndarray,
    Q_dval: np.ndarray,
    Q_ddval: np.ndarray,
    Q_offsets: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    weights: np.ndarray,
    mom: np.ndarray,
    n_active: int,
    Q_patch_area: np.ndarray,
    order: int,
    out_aperture: np.ndarray,
) -> None:
    """``M_a`` of one map on a coarse aperture level with the moment
    correction (``moments.aperture_shear_terms``; float64).  ``g1``/``g2``
    hold the transported cell means, ``mom`` ``(n_appended, NM, 2)`` the
    normalised moments of this map, cells at ``row - n_active``."""
    n_patches = Q_offsets.shape[0] - 1
    S = _S
    c1 = 2.0
    for patch_idx in prange(n_patches):
        start = Q_offsets[patch_idx]
        stop = Q_offsets[patch_idx + 1]
        sum_w = 0.0
        sum_n = 0.0
        for i in range(start, stop):
            row = Q_inds[i]
            cell = row - n_active
            psi = np.float64(Q_phi[i])
            s = np.sin(psi)
            c = np.cos(psi)
            idd = np.float64(Q_inv_theta[i])
            qv = np.float64(Q_val[i])
            qd = np.float64(Q_dval[i])
            qdd = np.float64(Q_ddval[i])
            px, py, qx, qy = s, -c, -c, -s
            m_r = np.float64(g1[row])
            m_i = np.float64(g2[row])
            N_r = qv * m_r
            N_i = qv * m_i
            if order >= 1:
                dx_r = np.float64(mom[cell, 1, 0])
                dx_i = np.float64(mom[cell, 1, 1])
                dy_r = np.float64(mom[cell, 2, 0])
                dy_i = np.float64(mom[cell, 2, 1])
                Dp_r = dx_r * px + dy_r * py
                Dp_i = dx_i * px + dy_i * py
                Dq_r = dx_r * qx + dy_r * qy
                Dq_i = dx_i * qx + dy_i * qy
                f1 = -c1 * S * idd
                N_r += qv * (-Dp_i * f1) + qd * Dq_r
                N_i += qv * (Dp_r * f1) + qd * Dq_i
                if order >= 2:
                    xx_r = np.float64(mom[cell, 3, 0])
                    xx_i = np.float64(mom[cell, 3, 1])
                    xy_r = np.float64(mom[cell, 4, 0])
                    xy_i = np.float64(mom[cell, 4, 1])
                    yy_r = np.float64(mom[cell, 5, 0])
                    yy_i = np.float64(mom[cell, 5, 1])
                    Qpp_r = xx_r * px * px + xy_r * 2.0 * px * py + yy_r * py * py
                    Qpp_i = xx_i * px * px + xy_i * 2.0 * px * py + yy_i * py * py
                    qpm = qx * py + qy * px
                    Qqp_r = xx_r * qx * px + xy_r * qpm + yy_r * qy * py
                    Qqp_i = xx_i * qx * px + xy_i * qpm + yy_i * qy * py
                    Qqq_r = xx_r * qx * qx + xy_r * 2.0 * qx * qy + yy_r * qy * qy
                    Qqq_i = xx_i * qx * qx + xy_i * 2.0 * qx * qy + yy_i * qy * qy
                    f2 = -0.5 * c1 * c1 * idd * idd
                    f3 = c1 * S * idd * idd
                    f4 = -c1 * S * idd
                    N_r += (
                        qv * (Qpp_r * f2 - Qqp_i * f3)
                        + qd * (-Qqp_i * f4)
                        + 0.5 * qdd * Qqq_r
                        + 0.5 * qd * idd * Qpp_r
                    )
                    N_i += (
                        qv * (Qpp_i * f2 + Qqp_r * f3)
                        + qd * (Qqp_r * f4)
                        + 0.5 * qdd * Qqq_i
                        + 0.5 * qd * idd * Qpp_i
                    )
            c2 = c * c - s * s
            s2 = 2.0 * s * c
            term = -(N_r * c2 - N_i * s2)
            weight = np.float64(weights[row])
            sum_w += weight
            sum_n += weight * term
        if sum_w == 0.0:
            out_aperture[patch_idx] = 0.0
        else:
            out_aperture[patch_idx] = Q_patch_area[patch_idx] * sum_n / sum_w


@njit(fastmath=True, parallel=True, cache=True)
def _cpu_aperture_density_moments_kernel(
    Q_inds: np.ndarray,
    Q_phi: np.ndarray,
    Q_inv_theta: np.ndarray,
    Q_val: np.ndarray,
    Q_dval: np.ndarray,
    Q_ddval: np.ndarray,
    Q_offsets: np.ndarray,
    map_values: np.ndarray,
    weights: np.ndarray,
    mom: np.ndarray,
    n_active: int,
    Q_patch_area: np.ndarray,
    order: int,
    out_aperture: np.ndarray,
) -> None:
    """``M_g`` with a filter U of one map on a coarse aperture level with the
    moment correction (``moments.aperture_scalar_terms``; float64)."""
    n_patches = Q_offsets.shape[0] - 1
    for patch_idx in prange(n_patches):
        start = Q_offsets[patch_idx]
        stop = Q_offsets[patch_idx + 1]
        sum_w = 0.0
        sum_n = 0.0
        for i in range(start, stop):
            row = Q_inds[i]
            cell = row - n_active
            psi = np.float64(Q_phi[i])
            s = np.sin(psi)
            c = np.cos(psi)
            idd = np.float64(Q_inv_theta[i])
            uv = np.float64(Q_val[i])
            ud = np.float64(Q_dval[i])
            udd = np.float64(Q_ddval[i])
            px, py, qx, qy = s, -c, -c, -s
            N = uv * np.float64(map_values[row])
            if order >= 1:
                Dq = np.float64(mom[cell, 1, 0]) * qx + np.float64(mom[cell, 2, 0]) * qy
                N += ud * Dq
                if order >= 2:
                    xx = np.float64(mom[cell, 3, 0])
                    xy = np.float64(mom[cell, 4, 0])
                    yy = np.float64(mom[cell, 5, 0])
                    Qpp = xx * px * px + xy * 2.0 * px * py + yy * py * py
                    Qqq = xx * qx * qx + xy * 2.0 * qx * qy + yy * qy * qy
                    N += 0.5 * udd * Qqq + 0.5 * ud * idd * Qpp
            weight = np.float64(weights[row])
            sum_w += weight
            sum_n += weight * N
        if sum_w == 0.0:
            out_aperture[patch_idx] = 0.0
        else:
            out_aperture[patch_idx] = Q_patch_area[patch_idx] * sum_n / sum_w


# ── GPU wrappers ─────────────────────────────────────────────────────────

_MOMENT_STRIDED = ("w_src", "v_src")


def _build_cupy_moment_level_kernel(module: Any) -> Any:
    """Builder for ``gpu_moment_level`` (``cuda/moment_rows.cu``): the
    normalised moments of one level straight from the pixel rows.

    Returns ``False`` without a raw compiler so the caller falls back to the
    sparse path.  ``w_src`` / ``v_src`` are the row-buffer descriptors of the
    degrade kernels (array, base, lead stride, component stride, row stride).
    """
    build = KernelFile(
        module, "moment_rows.cu", "Cell moments", strided=_MOMENT_STRIDED
    )

    def _cupy_moment_level_kernel(
        indptr: Any,
        indices: Any,
        cell_geom: Any,
        pix_xyz: Any,
        w_src: Tuple[Any, int, int, int, int],
        v_src: Tuple[Any, int, int, int, int],
        kind: int,
        nm: int,
        out_mom: Any,
        out_w: Any,
    ) -> bool:
        if not _has_raw_cuda_compiler(module):
            return False
        n_cells = int(indptr.shape[0] - 1)
        n_lead = int(out_w.shape[1]) if out_w.ndim > 1 else 1
        if n_cells <= 0:
            return True
        nnz = int(indices.size)
        mean_children = max(1.0, nnz / max(1, n_cells))
        seg = 1
        while seg < 32 and seg < mean_children:
            seg *= 2
        kernel = build(
            "gpu_moment_level",
            (
                c_real(w_src[0].dtype),
                c_real(pix_xyz.dtype),
                c_real(out_mom.dtype),
                int(kind),
                int(nm),
                seg,
            ),
        )
        if kernel is None:
            return False
        threads = 256
        groups = threads // seg
        total = n_cells * n_lead
        blocks = (total + groups - 1) // groups
        w_arr, w_base, w_lead, _w_comp, w_row = w_src
        v_arr, v_base, v_lead, v_comp, v_row = v_src
        kernel(
            (max(1, blocks),),
            (threads,),
            (
                indptr,
                indices,
                cell_geom,
                pix_xyz,
                w_arr,
                w_base,
                w_lead,
                w_row,
                v_arr,
                v_base,
                v_lead,
                v_comp,
                v_row,
                out_mom,
                out_w,
                n_cells,
                n_lead,
            ),
        )
        return True

    return _cupy_moment_level_kernel


def _build_cupy_xipm_moment_kernels(module: Any) -> Any:
    """Builder for the moment-corrected xi+/xi- tiles of the coarse bins
    (``gpu_tiled_tomo_reduce_xipm_mom`` / ``gpu_tiled_packed_reduce_xipm_mom``
    in ``tomo_vectorized_xipm.cu``).  Same output contract as the plain
    tiles; the launch covers every bin, the kernel skips empty ones.
    """
    build = KernelFile(module, "tomo_vectorized_xipm.cu", "Moment ξ±")

    def _launch(
        shear_map: Any,
        weights: Any,
        mom: Any,
        n_active: int,
        geometry: Tuple[Any, ...],
        bin_offsets: Any,
        comb_i: Any,
        comb_j: Any,
        order: int,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        """``geometry``: ``(ind_i, ind_j, phi_i, phi_j, inv_d, mom_base)``
        unpacked, or ``(pairs, row_base, d_codes, d_lo, d_scale, mom_base,
        cell_of_row)`` packed."""
        nzbins = int(shear_map.shape[1])
        if not _has_raw_cuda_compiler(module):
            return False
        layout = _combination_layout(comb_i, comb_j, "triangle", nzbins)
        if (
            layout is None
            or layout.mode != "triangle"
            or layout.rows is not None
            or 3 * layout.canonical_ncomb > _MAX_TILED_ACCUMULATORS
        ):
            return False
        map_c_type = c_real(weights.dtype)
        acc_c_type = c_real(out_num.dtype)
        nbins_total = int(bin_offsets.shape[0] - 1)
        packed = len(geometry) == 7
        if packed:
            pairs, row_base, d_codes, d_lo, d_scale, mom_base, cell_of_row = geometry
            kernel = build(
                "gpu_tiled_packed_reduce_xipm_mom",
                (map_c_type, nzbins, int(order), acc_c_type),
            )
            if kernel is None:
                return False
            kernel(
                (max(1, nbins_total), 1, 1),
                (256,),
                (
                    shear_map,
                    weights,
                    mom,
                    pairs,
                    d_codes,
                    d_lo,
                    d_scale,
                    bin_offsets,
                    row_base,
                    mom_base,
                    cell_of_row,
                    out_num,
                    out_den,
                    nbins_total,
                ),
            )
            return True
        ind_i, ind_j, phi_i, phi_j, inv_d, mom_base = geometry
        kernel = build(
            "gpu_tiled_tomo_reduce_xipm_mom",
            (
                map_c_type,
                c_real(phi_i.dtype),
                nzbins,
                int(order),
                c_index(ind_i.dtype),
                acc_c_type,
            ),
        )
        if kernel is None:
            return False
        kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (
                shear_map,
                weights,
                mom,
                ind_i,
                ind_j,
                phi_i,
                phi_j,
                inv_d,
                bin_offsets,
                mom_base,
                n_active,
                out_num,
                out_den,
                nbins_total,
            ),
        )
        return True

    return _launch


def _build_cupy_ds_moment_kernels(module: Any) -> Any:
    """Builder for the moment-corrected gamma_t tiles of the coarse bins
    (``density_shear_tomo_vectorized.cu``)."""
    build = KernelFile(module, "density_shear_tomo_vectorized.cu", "Moment γ_t")

    def _launch(
        density_map: Any,
        shear_map: Any,
        lens_weights: Any,
        source_weights: Any,
        mom_lens: Any,
        mom_src: Any,
        n_active: int,
        geometry: Tuple[Any, ...],
        bin_offsets: Any,
        comb_i: Any,
        comb_j: Any,
        order: int,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        nlens = int(density_map.shape[1])
        nsrc = int(shear_map.shape[1])
        if not _has_raw_cuda_compiler(module):
            return False
        layout = _combination_layout(comb_i, comb_j, "cartesian", nlens, nsrc)
        if (
            layout is None
            or layout.rows is not None
            or 2 * layout.canonical_ncomb > _MAX_TILED_ACCUMULATORS
        ):
            return False
        map_c_type = c_real(lens_weights.dtype)
        acc_c_type = c_real(out_num.dtype)
        nbins_total = int(bin_offsets.shape[0] - 1)
        packed = len(geometry) == 7
        if packed:
            pairs, row_base, d_codes, d_lo, d_scale, mom_base, cell_of_row = geometry
            kernel = build(
                "gpu_tiled_packed_reduce_ds_mom",
                (map_c_type, nlens, nsrc, int(order), acc_c_type),
            )
            if kernel is None:
                return False
            kernel(
                (max(1, nbins_total), 1, 1),
                (256,),
                (
                    density_map,
                    shear_map,
                    lens_weights,
                    source_weights,
                    mom_lens,
                    mom_src,
                    pairs,
                    d_codes,
                    d_lo,
                    d_scale,
                    bin_offsets,
                    row_base,
                    mom_base,
                    cell_of_row,
                    out_num,
                    out_den,
                    nbins_total,
                ),
            )
            return True
        ind_i, ind_j, phi_i, phi_j, inv_d, mom_base = geometry
        kernel = build(
            "gpu_tiled_tomo_reduce_ds_mom",
            (
                map_c_type,
                c_real(phi_i.dtype),
                nlens,
                nsrc,
                int(order),
                c_index(ind_i.dtype),
                acc_c_type,
            ),
        )
        if kernel is None:
            return False
        kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (
                density_map,
                shear_map,
                lens_weights,
                source_weights,
                mom_lens,
                mom_src,
                ind_i,
                ind_j,
                phi_i,
                phi_j,
                inv_d,
                bin_offsets,
                mom_base,
                n_active,
                out_num,
                out_den,
                nbins_total,
            ),
        )
        return True

    return _launch


def _build_cupy_3x2pt_moment_kernel(module: Any) -> Any:
    """Builder for the moment-corrected multi-statistic tile of the coarse
    bins (``gpu_tiled_{tomo,packed}_reduce_3x2pt_mom`` in
    ``tomo_tiled_3x2pt.cu``): xi+- and xi_t with moments, xi_g plain, in one
    walk.  Returns ``False`` when the accumulators do not fit the budget;
    the orchestrator then launches the standalone moment tiles."""
    build = KernelFile(module, "tomo_tiled_3x2pt.cu", "Moment 3x2pt")

    def _launch(
        density_map: Any,
        shear_map: Any,
        density_w: Any,
        shear_w: Any,
        mom_lens: Any,
        mom_src: Any,
        n_active: int,
        geometry: Tuple[Any, ...],
        bin_offsets: Any,
        ss_comb: Tuple[Any, Any],
        dd_comb: Tuple[Any, Any],
        ds_comb: Tuple[Any, Any],
        order: int,
        max_accumulators: int,
        out_xipm_num: Any,
        out_xipm_den: Any,
        out_xig_num: Any,
        out_xig_den: Any,
        out_xit_num: Any,
        out_xit_den: Any,
    ) -> bool:
        if not _has_raw_cuda_compiler(module):
            return False
        n_density = int(density_map.shape[1])
        n_shear = int(shear_map.shape[1])
        ss = _combination_layout(ss_comb[0], ss_comb[1], "triangle", n_shear)
        dd = _combination_layout(dd_comb[0], dd_comb[1], "triangle", n_density)
        ds = _combination_layout(
            ds_comb[0], ds_comb[1], "cartesian", n_density, n_shear
        )
        if (
            ss is None
            or dd is None
            or ds is None
            or ss.mode != "triangle"
            or ss.rows is not None
            or dd.rows is not None
            or ds.rows is not None
        ):
            return False
        n_acc = 3 * ss.canonical_ncomb + 2 * dd.canonical_ncomb + 2 * ds.canonical_ncomb
        if n_acc > max_accumulators:
            return False
        map_c_type = c_real(density_map.dtype)
        acc_c_type = c_real(out_xipm_num.dtype)
        auto_only = 1 if dd.mode == "auto" else 0
        nbins_total = int(bin_offsets.shape[0] - 1)
        packed = len(geometry) == 7
        if packed:
            pairs, row_base, d_codes, d_lo, d_scale, mom_base, cell_of_row = geometry
            kernel = build(
                "gpu_tiled_packed_reduce_3x2pt_mom",
                (map_c_type, n_density, n_shear, auto_only, int(order), acc_c_type),
            )
            if kernel is None:
                return False
            kernel(
                (max(1, nbins_total), 1, 1),
                (256,),
                (
                    density_map,
                    shear_map,
                    density_w,
                    shear_w,
                    mom_lens,
                    mom_src,
                    pairs,
                    d_codes,
                    d_lo,
                    d_scale,
                    bin_offsets,
                    row_base,
                    mom_base,
                    cell_of_row,
                    out_xipm_num,
                    out_xipm_den,
                    out_xig_num,
                    out_xig_den,
                    out_xit_num,
                    out_xit_den,
                    nbins_total,
                ),
            )
            return True
        ind_i, ind_j, phi_i, phi_j, inv_d, mom_base = geometry
        kernel = build(
            "gpu_tiled_tomo_reduce_3x2pt_mom",
            (
                map_c_type,
                c_real(phi_i.dtype),
                n_density,
                n_shear,
                auto_only,
                int(order),
                c_index(ind_i.dtype),
                acc_c_type,
            ),
        )
        if kernel is None:
            return False
        kernel(
            (max(1, nbins_total), 1, 1),
            (256,),
            (
                density_map,
                shear_map,
                density_w,
                shear_w,
                mom_lens,
                mom_src,
                ind_i,
                ind_j,
                phi_i,
                phi_j,
                inv_d,
                bin_offsets,
                mom_base,
                n_active,
                out_xipm_num,
                out_xipm_den,
                out_xig_num,
                out_xig_den,
                out_xit_num,
                out_xit_den,
                nbins_total,
            ),
        )
        return True

    return _launch


_APERTURE_STRIDED = ("g1", "g2", "weights", "values")


def _build_cupy_aperture_moments_kernels(module: Any) -> Any:
    """Builder for the aperture statistics of moment-carrying aperture cells
    (``gpu_aperture_shear_tomo_mom`` / ``gpu_aperture_density_tomo_mom`` in
    ``aperture_tomo.cu``; one block per patch, the bins inside it).  Returns
    a callable ``launch(kind, ...)`` with ``kind`` ``"shear"`` or
    ``"scalar"``; ``False`` without a raw compiler or beyond 16 bins."""
    build = KernelFile(
        module, "aperture_tomo.cu", "Aperture moments", strided=_APERTURE_STRIDED
    )

    def _strides(arr: Any) -> Tuple[int, int]:
        return int(arr.strides[0] // arr.itemsize), int(arr.strides[1] // arr.itemsize)

    def _launch(
        kind: str,
        values: Tuple[Any, ...],
        weights: Any,
        mom: Any,
        n_active: int,
        geometry: Tuple[Any, ...],
        order: int,
        out_num: Any,
        out_den: Any,
    ) -> bool:
        """``values``: ``(g1, g2)`` 2D views for shear, ``(v,)`` for a
        scalar; ``geometry``: ``(q_inds, q_phi, q_inv_theta, q_val, q_dval,
        q_ddval, q_offsets, q_patch_area)``."""
        if not _has_raw_cuda_compiler(module):
            return False
        ntomo = int(weights.shape[0])
        if ntomo <= 0 or ntomo > 16:
            return False
        q_inds, q_phi, q_inv_theta, q_val, q_dval, q_ddval, q_offsets, q_area = geometry
        npatches = int(q_offsets.shape[0] - 1)
        targs = (c_real(weights.dtype), c_real(q_phi.dtype), ntomo, int(order))
        w_stride, w_elem = _strides(weights)
        geom_args = (
            mom,
            n_active,
            q_inds,
            q_phi,
            q_inv_theta,
            q_val,
            q_dval,
            q_ddval,
            q_offsets,
            q_area,
            out_num,
            out_den,
            npatches,
        )
        if kind == "shear":
            g1, g2 = values
            g1_stride, g1_elem = _strides(g1)
            if _strides(g2) != (g1_stride, g1_elem):
                g1 = module.ascontiguousarray(g1)
                g2 = module.ascontiguousarray(g2)
                g1_stride, g1_elem = int(g1.shape[1]), 1
            kernel = build("gpu_aperture_shear_tomo_mom", targs)
            if kernel is None:
                return False
            kernel(
                (max(1, npatches), 1, 1),
                (256,),
                (g1, g2, g1_stride, g1_elem, weights, w_stride, w_elem, *geom_args),
            )
            return True
        (v,) = values
        v_stride, v_elem = _strides(v)
        kernel = build("gpu_aperture_density_tomo_mom", targs)
        if kernel is None:
            return False
        kernel(
            (max(1, npatches), 1, 1),
            (256,),
            (v, v_stride, v_elem, weights, w_stride, w_elem, *geom_args),
        )
        return True

    return _launch


def _build_cupy_3x2pt_aperture_moments_kernel(module: Any) -> Any:
    """Builder for ``gpu_3x2pt_tomo_aperture_mom`` (``tomo_fused_3x2pt.cu``):
    the aperture sections of the fused path with moment-carrying aperture
    cells, AoS inputs, ACC accumulation, one launch per section."""
    build = KernelFile(module, "tomo_fused_3x2pt.cu", "Fused aperture moments")

    def _launch(
        density_map: Any,
        shear_map: Any,
        density_weights: Any,
        shear_weights: Any,
        mom_lens: Any,
        mom_src: Any,
        n_active: int,
        q_geometry: Tuple[Any, ...],
        k_geometry: Tuple[Any, ...],
        order: int,
        out_ma_num: Any,
        out_ma_den: Any,
        out_mg_num: Any,
        out_mg_den: Any,
    ) -> bool:
        if not _has_raw_cuda_compiler(module):
            return False
        n_density = int(density_map.shape[1])
        n_shear = int(shear_map.shape[1])
        if max(n_density, n_shear) > 16:
            return False
        q_inds, q_phi, q_inv_theta, q_val, q_dval, q_ddval, q_offsets, q_area = (
            q_geometry
        )
        k_inds, k_phi, k_inv_theta, k_val, k_dval, k_ddval, k_offsets, k_area = (
            k_geometry
        )
        npatches = int(q_offsets.shape[0] - 1)
        kernel = build(
            "gpu_3x2pt_tomo_aperture_mom",
            (
                c_real(density_map.dtype),
                c_real(q_phi.dtype),
                n_density,
                n_shear,
                c_real(out_ma_num.dtype),
                int(order),
            ),
        )
        if kernel is None:
            return False
        args = (
            density_map,
            shear_map,
            density_weights,
            shear_weights,
            mom_lens,
            mom_src,
            n_active,
            npatches,
            q_inds,
            q_phi,
            q_inv_theta,
            q_val,
            q_dval,
            q_ddval,
            q_offsets,
            q_area,
            k_inds,
            k_phi,
            k_inv_theta,
            k_val,
            k_dval,
            k_ddval,
            k_offsets,
            k_area,
            out_ma_num,
            out_ma_den,
            out_mg_num,
            out_mg_den,
        )
        for section in (0, 1):
            kernel((max(1, npatches), 1, 1), (256,), args + (section,))
        return True

    return _launch

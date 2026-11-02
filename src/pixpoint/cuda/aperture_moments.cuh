/*
 * aperture_moments.cuh -- Per-cell terms of the moment-carrying aperture
 *                         statistics (cell_moments with aperture_nside).
 *
 * Shared by aperture_tomo.cu and tomo_fused_3x2pt.cu; the expansion is
 * documented in the former's "Moment-carrying aperture cells" section and in
 * pixpoint/moments.py.  Requires common.cuh (sincos_t) to be included first.
 */

template<typename T, int MOM>
__device__ __forceinline__ T aperture_shear_mom_term(
    const T* mom_cell, T m_r, T m_i, T s, T c, T idd, T qv, T qd, T qdd)
{
    const T S = (T)-1;
    const T px = s, py = -c, qx = -c, qy = -s;
    T N_r = qv * m_r, N_i = qv * m_i;
    if (MOM >= 1) {
        const T dx_r = mom_cell[2], dx_i = mom_cell[3];
        const T dy_r = mom_cell[4], dy_i = mom_cell[5];
        const T Dp_r = dx_r * px + dy_r * py, Dp_i = dx_i * px + dy_i * py;
        const T Dq_r = dx_r * qx + dy_r * qy, Dq_i = dx_i * qx + dy_i * qy;
        const T f1 = -(T)2 * S * idd;              /* N += qv * i Dp f1 */
        N_r += qv * (-Dp_i * f1) + qd * Dq_r;
        N_i += qv * (Dp_r * f1) + qd * Dq_i;
        if (MOM >= 2) {
            const T xx_r = mom_cell[6], xx_i = mom_cell[7];
            const T xy_r = mom_cell[8], xy_i = mom_cell[9];
            const T yy_r = mom_cell[10], yy_i = mom_cell[11];
            const T Qpp_r = xx_r * px * px + xy_r * (T)2 * px * py + yy_r * py * py;
            const T Qpp_i = xx_i * px * px + xy_i * (T)2 * px * py + yy_i * py * py;
            const T qpm = qx * py + qy * px;
            const T Qqp_r = xx_r * qx * px + xy_r * qpm + yy_r * qy * py;
            const T Qqp_i = xx_i * qx * px + xy_i * qpm + yy_i * qy * py;
            const T Qqq_r = xx_r * qx * qx + xy_r * (T)2 * qx * qy + yy_r * qy * qy;
            const T Qqq_i = xx_i * qx * qx + xy_i * (T)2 * qx * qy + yy_i * qy * qy;
            const T f2 = -(T)2 * idd * idd;             /* qv * Qpp f2 */
            const T f3 = (T)2 * S * idd * idd;          /* qv * i Qqp f3 */
            const T f4 = -(T)2 * S * idd;               /* qd * i Qqp f4 */
            N_r += qv * (Qpp_r * f2 - Qqp_i * f3) + qd * (-Qqp_i * f4)
                 + (T)0.5 * qdd * Qqq_r + (T)0.5 * qd * idd * Qpp_r;
            N_i += qv * (Qpp_i * f2 + Qqp_r * f3) + qd * (Qqp_r * f4)
                 + (T)0.5 * qdd * Qqq_i + (T)0.5 * qd * idd * Qpp_i;
        }
    }
    /* -Re(N e^{2i psi}) */
    const T c2 = c * c - s * s, s2 = (T)2 * s * c;
    return -(N_r * c2 - N_i * s2);
}

template<typename T, int MOM>
__device__ __forceinline__ T aperture_scalar_mom_term(
    const T* mom_cell, T m, T s, T c, T idd, T uv, T ud, T udd)
{
    const T px = s, py = -c, qx = -c, qy = -s;
    T N = uv * m;
    if (MOM >= 1) {
        const T Dq = mom_cell[2] * qx + mom_cell[4] * qy;
        N += ud * Dq;
        if (MOM >= 2) {
            const T xx = mom_cell[6], xy = mom_cell[8], yy = mom_cell[10];
            const T Qpp = xx * px * px + xy * (T)2 * px * py + yy * py * py;
            const T Qqq = xx * qx * qx + xy * (T)2 * qx * qy + yy * qy * qy;
            N += (T)0.5 * udd * Qqq + (T)0.5 * ud * idd * Qpp;
        }
    }
    return N;
}


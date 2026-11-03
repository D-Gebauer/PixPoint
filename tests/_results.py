"""Test helper: named fields of a result, as a tuple, in the order asked."""


def fields_of(result, *names):
    return tuple(getattr(result, n) for n in names)


# Which sample sits at the centre and in the annulus of each zeta estimator,
# and whether the two share one tomography (triangle layout) or cross two
# samples (centre-major layout over the annulus combinations).
ZETA_DEFINITIONS = {
    "zeta_g_plus": ("M_g", "xi_p", False),
    "zeta_g_minus": ("M_g", "xi_m", False),
    "zeta_a_plus": ("M_a", "xi_p", True),
    "zeta_a_minus": ("M_a", "xi_m", True),
    "zeta_g_g": ("M_g", "xi_g", True),
    "zeta_a_g": ("M_a", "xi_g", False),
    "zeta_g_t": ("M_g", "xi_t", False),
    "zeta_a_t": ("M_a", "xi_t", False),
}

_ANNULUS_COMBINATIONS = {
    "xi_p": "gg_combinations",
    "xi_m": "gg_combinations",
    "xi_g": "kk_combinations",
    "xi_t": "kg_combinations",
}


def zeta_reference(result, name):
    """An i3PCF estimator of an ``IntegratedResult`` by an explicit loop over
    its rows: the covariance over patches of the central aperture and the
    annulus 2PCF, <C A> - <C><A>, rows in the documented order
    (``Integrated3PCF.zeta_triplets`` / ``zeta_cross_triplets``)."""
    import itertools

    import numpy as np

    central_name, annulus_name, triangle = ZETA_DEFINITIONS[name]
    central = np.asarray(getattr(result, central_name), dtype=np.float64)
    annulus = np.asarray(getattr(result, annulus_name), dtype=np.float64)
    combinations = list(getattr(result, _ANNULUS_COMBINATIONS[annulus_name]))
    if triangle:
        rows = [
            (zc, combinations.index((z2, z3)))
            for zc, z2, z3 in itertools.combinations_with_replacement(
                range(central.shape[0]), 3
            )
        ]
    else:
        rows = [
            (zc, a) for zc in range(central.shape[0]) for a in range(len(combinations))
        ]
    out = []
    for zc, a in rows:
        c = central[zc]  # (n_patches,)
        x = annulus[a]  # (n_patches, nbins)
        out.append((c[:, None] * x).mean(axis=0) - c.mean() * x.mean(axis=0))
    return np.array(out)

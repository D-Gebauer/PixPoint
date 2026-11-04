"""
PixPoint: GPU/CPU measurement of integrated, two- and three-point correlations.

Three estimators on HEALPix maps, named after the statistic they deliver:
:class:`Integrated3PCF` (per-patch aperture masses and 2PCFs, the i3PCF
ingredients), :class:`Global2PCF` (footprint-wide 2PCFs),
:class:`Global3PCF` (footprint-wide 3PCF multipoles) and :class:`OnePoint`
(PDFs and moments of aperture fields: ``g`` for the aperture mass, ``k`` for a
filtered scalar field).  Methods are TreeCorr
field codes (``g`` = shear, ``k`` = scalar field): ``gg``, ``kk``, ``kg``,
``three_by_two_pt``, ``ggg``, ``kkk``, ``kgg``, ``kkg``.  Every call returns a
labelled result (:class:`IntegratedResult`, :class:`TwoPointResult`,
:class:`ThreePointResult`, :class:`OnePointResult`).
"""

from .correlation_helpers import (
    Q_crittenden,
    Q_schneider,
    U_crittenden,
    U_schneider,
    U_tophat,
    calculate_all_zetas,
    zeta_a_g,
    zeta_a_minus,
    zeta_a_plus,
    zeta_a_t,
    zeta_g_g,
    zeta_g_minus,
    zeta_g_plus,
    zeta_g_t,
)
from .correlations import Integrated3PCF
from .global_2pcf import Global2PCF
from .global_3pcf import Global3PCF
from .onepoint import OnePoint
from .pipeline import MapFileLoader, MapLoader
from .result_writer import OnePointWriter, ResultWriter, result_arrays
from .results import (
    Component,
    IntegratedResult,
    OnePointResult,
    ThreePointResult,
    TwoPointResult,
)
from .treecode import DEFAULT_RESOLUTION_FACTOR
from .utils import pixel2RaDec, select_patch_centers
from .zeta_writer import ZetaWriter

__version__ = "8.1.0"
__author__ = "David Gebauer"
__email__ = "git@gebauer.ai"

__all__ = [
    "Component",
    "Global2PCF",
    "Global3PCF",
    "Integrated3PCF",
    "IntegratedResult",
    "OnePoint",
    "OnePointResult",
    "OnePointWriter",
    "ResultWriter",
    "result_arrays",
    "ThreePointResult",
    "TwoPointResult",
    "DEFAULT_RESOLUTION_FACTOR",
    "MapLoader",
    "MapFileLoader",
    "Q_crittenden",
    "Q_schneider",
    "U_crittenden",
    "U_schneider",
    "U_tophat",
    "calculate_all_zetas",
    "zeta_g_plus",
    "zeta_g_minus",
    "zeta_a_plus",
    "zeta_a_minus",
    "zeta_g_g",
    "zeta_a_g",
    "zeta_g_t",
    "zeta_a_t",
    "pixel2RaDec",
    "select_patch_centers",
    "ZetaWriter",
]

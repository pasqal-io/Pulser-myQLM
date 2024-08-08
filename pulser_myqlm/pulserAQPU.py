"""Legacy import path."""

import warnings

from pulser_myqlm.fresnel_qpu import FresnelQPU  # noqa F401
from pulser_myqlm.ising_aqpu import IsingAQPU  # noqa F401

warnings.warn(
    "This module is deprecated. You should import from"
    + "pulser_myqlm.frensel_qpu or pulser_myqlm.ising_aqpu"
    + "instead. Support for imports from this module will stop"
    + "in version 1.0.0 of this library.",
    DeprecationWarning,
    stacklevel=2,
)

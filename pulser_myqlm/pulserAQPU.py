"""Legacy import path."""

import warnings

from pulser_myqlm.fresnel_qpu import FresnelQPU
from pulser_myqlm.ising_aqpu import IsingAQPU

__all__ = ["FresnelQPU", "IsingAQPU"]

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        (
            "This module is deprecated. Please import from "
            "pulser_myqlm.fresnel_qpu or pulser_myqlm.ising_aqpu instead. "
            "Support for imports from this module will be removed in "
            "version 1.0.0 of this library."
        ),
        DeprecationWarning,
        stacklevel=2,
    )

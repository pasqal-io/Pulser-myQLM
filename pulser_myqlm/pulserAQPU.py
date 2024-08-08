import warnings

from pulser_myqlm.fresnel_qpu import FresnelQPU
from pulser_myqlm.ising_aqpu import IsingAQPU


warnings.warn(
    "This module is deprecated. Do not import from here",
    DeprecationWarning,
    stacklevel=2,
)
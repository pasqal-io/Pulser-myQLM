"""Example of a conversion of a Pulser Sequence into MyQLM."""
import numpy as np
from pulser import Pulse, Register, Sequence
from pulser.devices import MockDevice
from pulser.waveforms import CustomWaveform

from pulser_myqlm import IsingAQPU

# Pulser Sequence (find more at https://pulser.readthedocs.io/)

device = MockDevice
register = Register.square(2, 4, None)
seq = Sequence(register, device)
seq.declare_channel("ryd_glob", "rydberg_global")
seq.add(
    Pulse(
        CustomWaveform([ti for ti in range(16)]),
        CustomWaveform([1 - ti for ti in range(16)]),
        0,
    ),
    "ryd_glob",
)
seq.add(Pulse.ConstantPulse(20, 1, 0, 0), "ryd_glob")
seq.add(Pulse.ConstantPulse(20, 1, 0, np.pi / 2), "ryd_glob")

# Convert to MyQLM Job
IsingAQPU.convert_sequence_to_job(seq)

# Convert to MyQLM Schedule
IsingAQPU.convert_sequence_to_schedule(seq)

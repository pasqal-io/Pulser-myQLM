"""Example of a conversion of a Pulser Sequence into MyQLM."""

import numpy as np
from pulser import Pulse, Register, Sequence
from pulser.devices import AnalogDevice
from pulser.waveforms import CustomWaveform

from pulser_myqlm import IsingAQPU

# Pulser Sequence (find more at https://pulser.readthedocs.io/)

device = AnalogDevice
register = Register.square(2, 5, None)
seq = Sequence(register, device)
seq.declare_channel("ryd_glob", "rydberg_global")
duration = 100
seq.add(
    Pulse(
        CustomWaveform([ti / duration for ti in range(duration)]),
        CustomWaveform([1 - ti / duration for ti in range(duration)]),
        0,
    ),
    "ryd_glob",
)
seq.add(Pulse.ConstantPulse(20, 1, 0, 0), "ryd_glob")
seq.add(Pulse.ConstantPulse(20, 1, 0, np.pi / 2), "ryd_glob")

# Convert the Sequence to a Job
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=1000, modulation=True)

# Simulate the Job using pulser_simulation
aqpu = IsingAQPU.from_sequence(seq, qpu=None)
results = aqpu.submit(job)

# Simulate the Job using AnalogQPU
try:
    from qlmaas.qpus import AnalogQPU

    analog_qpu = AnalogQPU()
    aqpu = IsingAQPU.from_sequence(seq, qpu=analog_qpu)
    results = aqpu.submit(job)
except ImportError:
    print("Can't import AnalogQPU, check connection to Qaptiva Access.")

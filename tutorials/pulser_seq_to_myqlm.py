"""Example of a conversion of a Pulser Sequence into MyQLM."""

import numpy as np
from pulser import Pulse, Register, Sequence
from pulser.devices import AnalogDevice
from pulser.waveforms import CustomWaveform
from pulser_simulation import QutipEmulator

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

# Draw the Sequence
seq.draw(draw_phase_curve=True)

# Simulate the Sequence using Pulser
sim = QutipEmulator.from_sequence(seq, with_modulation=True)
res = sim.run().sample_final_state(2000)
print("Pulser Result obtained with pulser_simulation for 2000 samples:")
print(res, "\n")
print("Converted into MyQLM Result:")
print(IsingAQPU.convert_samples_to_result(res), "\n")
print(
    "Expressed as a dictionary of (state: probability): ",
    {
        sample.state: sample.probability
        for sample in IsingAQPU.convert_samples_to_result(res)
    },
    "\n",
)

# Convert the Sequence to a Job
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=0, modulation=True)

# Simulate the Job using pulser_simulation
aqpu = IsingAQPU.from_sequence(seq, qpu=None)
result = aqpu.submit(job)
print(
    "MyQLM Result obtained using IsingAQPU with pulser-simulation with "
    "default number of samples (2000):"
)
print(result, "\n")
print(
    "Expressed as a dictionary of (state: probability): ",
    {sample.state: sample.probability for sample in result},
    "\n",
)

print("Converted into a Pulser Result:")
print(IsingAQPU.convert_result_to_samples(result), "\n")

# Simulate the Job using AnalogQPU
try:
    from qlmaas.qpus import AnalogQPU

    analog_qpu = AnalogQPU()
    aqpu = IsingAQPU.from_sequence(seq, qpu=analog_qpu)
    results = aqpu.submit(job)
    # Display the results once they have run on AnalogQPU
    print("Results obtained with AnalogQPU: ", results.join())
    print(
        "Expressed as a dictionary of (state: probability): ",
        {sample.state: sample.probability for sample in results},
        "\n",
    )
except ImportError:
    print("Can't import AnalogQPU, check connection to Qaptiva Access.")

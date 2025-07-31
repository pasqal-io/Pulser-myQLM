"""Script to connect to a Remote FresnelQPU and prepare an AntiFerroMagnetic state."""

import matplotlib.pyplot as plt
import numpy as np
from pulser import Pulse, Sequence, InterpolatedWaveform, QPUBackend
from pulser.backend.remote import JobParams

from pulser_myqlm import PulserQLMConnection

# Connect to the QPU
conn = PulserQLMConnection()

# Get the Device implemented by the QPU from the QPU specs
FRESNEL_DEVICE = conn.fetch_available_devices()["qat.qpus:PasqalQPU"]
print("Using the Device:", "\n")
FRESNEL_DEVICE.print_specs()

# Simulation
reg_layout = FRESNEL_DEVICE.calibrated_register_layouts[
    "TriangularLatticeLayout(61, 5.0µm)"
]
Omega_max = 0.8 * 2 * FRESNEL_DEVICE.rabi_from_blockade(5.0)  # Spacing between atoms

U = Omega_max / 2.0

T = 2  # us
params = [T] + (
    U * np.array([0.16768532, 0.2, 0.2, -1.0, -0.54656236, 0.05762063, 0.3673201, 1.0])
).tolist()

interpolated_pulse = Pulse(
    InterpolatedWaveform(
        T * 1000,
        U * np.array([1e-9, 0.16768532, 0.2, 0.2, 1e-9]),
        times=np.linspace(0, 1, 5),
    ),
    InterpolatedWaveform(
        T * 1000,
        U * np.array([-1.0, -0.54656236, 0.05762063, 0.3673201, 1.0]),
        times=np.linspace(0, 1, 5),
    ),
    0,
)

reg = reg_layout.define_register(21, 26, 35, 39, 34, 25)
N_atoms = len(reg.qubits)
print("Contains ", N_atoms, "atoms")

# Creating the Sequence
seq = Sequence(reg, FRESNEL_DEVICE)
seq.declare_channel("ising", "rydberg_global")

seq.add(interpolated_pulse, "ising")

# Simulate the Sequence on the QPU
qpu = QPUBackend(seq, connection=conn)

remote_results = qpu.run([JobParams(runs=1000, variables=[])], wait=True)
count = remote_results.get_available_results().values()[0]

# Print the most interesting samples
print("Obtained samples:", count)

most_freq = {k: v for k, v in count.items() if v > 10 / 1000}
plt.bar(list(most_freq.keys()), list(most_freq.values()))
plt.xticks(rotation="vertical")
plt.show()

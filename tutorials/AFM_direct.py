"""Script to connect to a Remote FresnelQPU and prepare an AntiFerroMagnetic state."""

import matplotlib.pyplot as plt
import numpy as np
from pulser import Pulse, Sequence, InterpolatedWaveform
from pulser.devices import Device

from pulser_myqlm import IsingAQPU, FresnelQPU

# Connect to the QPU
PORT = 1234
IP = "127.0.0.1"  # TODO: Modify this IP
QPU = FresnelQPU(f"http://{IP}:{PORT}/api", version="v1")
print("qpu status:", QPU.is_operational)

# Get the Device implemented by the QPU from the QPU specs
FRESNEL_DEVICE = Device.from_abstract_repr(QPU.get_specs().description)
print("Using the Device:", "\n")
FRESNEL_DEVICE.print_specs()

# Simulation parameters
NBSHOTS = 0  # must be 0 for AnalogQPU
MODULATION = False  # Whether or not to use Modulated Sequence in the simulation
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
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=NBSHOTS, modulation=MODULATION)

results = QPU.submit(job)

# Print the most interesting samples
for sample in results:
    if sample.probability > 0.01:
        print(sample.state, sample.probability)


# Plot these results
def get_samples_from_result(result, n_qubits):
    """Converting the MyQLM Results into Pulser Samples."""
    samples = {}
    for sample in result.raw_data:
        if len(sample.state.bitstring) > n_qubits:
            raise ValueError(
                f"State {sample.state} is incompatible with number of qubits"
                f" declared {n_qubits}."
            )
        counts = sample.probability
        samples[sample.state.bitstring.zfill(n_qubits)] = counts
    return samples


count = get_samples_from_result(results, N_atoms)

most_freq = {k: v for k, v in count.items() if v > 10 / 1000}
plt.bar(list(most_freq.keys()), list(most_freq.values()))
plt.xticks(rotation="vertical")
plt.show()

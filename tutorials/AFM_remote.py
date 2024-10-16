"""Script to connect to a Remote FresnelQPU and prepare an AntiFerroMagnetic state."""

import matplotlib.pyplot as plt
import numpy as np
from pulser import Pulse, Sequence
from pulser.devices import Device
from pulser.waveforms import RampWaveform
from qat.qpus import RemoteQPU

from pulser_myqlm import FresnelQPU, IsingAQPU

# Connect to the QPU
PORT = 1234
IP = ""  # TODO: Modify this IP
QPU = RemoteQPU(PORT, IP)

# Get the device of the QPU
# TODO: Implement it with get_specs
FRESNEL_QPU = FresnelQPU(None)  # mimics QPU with pulser-simulation as backend
# Get the Device implemented by the QPU from the QPU specs
FRESNEL_DEVICE = Device.from_abstract_repr(FRESNEL_QPU.get_specs().description)
# Simulation parameters
NBSHOTS = 0  # must be 0 for AnalogQPU
MODULATION = False  # Whether or not to use Modulated Sequence in the simulation
# Simulation
reg_layout = FRESNEL_DEVICE.calibrated_register_layouts[
    "TriangularLatticeLayout(61, 5.0µm)"
]
Omega_max = 2 * FRESNEL_DEVICE.rabi_from_blockade(10.0)  # Spacing between atoms

U = Omega_max / 2.0

delta_0 = -6 * U
delta_f = 2 * U

t_rise = 2500
t_fall = 5000
t_sweep = (delta_f - delta_0) / (2 * np.pi) * 1000

R_interatomic = FRESNEL_DEVICE.rydberg_blockade_radius(U)

reg = reg_layout.define_register(22, 40, 48, 38, 20, 12)
N_atoms = len(reg.qubits)
print("Contains ", N_atoms, "atoms")
print(f"Interatomic Radius is: {R_interatomic}µm.")

rise = Pulse.ConstantDetuning(RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0)
sweep = Pulse.ConstantAmplitude(Omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.0)
fall = Pulse.ConstantDetuning(RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0)

# Creating the Sequence
seq = Sequence(reg, FRESNEL_DEVICE)
seq.declare_channel("ising", "rydberg_global")

seq.add(rise, "ising")
seq.add(sweep, "ising")
seq.add(fall, "ising")

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

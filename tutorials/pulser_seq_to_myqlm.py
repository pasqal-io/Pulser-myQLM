"""Example of a conversion of a Pulser Sequence into MyQLM."""

from threading import Thread

import numpy as np
from pulser import Pulse, Sequence
from pulser.devices import AnalogDevice
from pulser.waveforms import CustomWaveform
from pulser_simulation import QutipEmulator
from qat.qpus import RemoteQPU

from pulser_myqlm import FresnelQPU, IsingAQPU

# Pulser Sequence (find more at https://pulser.readthedocs.io/)
device = AnalogDevice
register = AnalogDevice.calibrated_register_layouts[
    "TriangularLatticeLayout(61, 5.0µm)"
].define_register(26, 35, 30)
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
# seq.draw(draw_phase_curve=True)

# Simulate the Sequence using Pulser
np.random.seed(123)
sim = QutipEmulator.from_sequence(seq, with_modulation=True)
res = sim.run().sample_final_state(2000)
# Convert the results into MyQLM
myqlm_res = IsingAQPU.convert_samples_to_result(res)
out_myqlm_res = {sample.state.__str__(): sample.probability for sample in myqlm_res}
print("Pulser Result obtained with pulser_simulation for 2000 samples:")
print(res, "\n")
print("Converted into MyQLM Result:")
print(myqlm_res, "\n")
print("Expressed as a dictionary of (state: probability): ")
print(out_myqlm_res, "\n")

exp_res = {"000": 1991, "100": 4, "010": 3, "001": 2}
exp_myqlm_res = {"|000>": 0.9955, "|001>": 0.001, "|010>": 0.0015, "|100>": 0.002}
assert res == exp_res
assert out_myqlm_res == exp_myqlm_res

# Convert the Sequence to a Job
job = IsingAQPU.convert_sequence_to_job(seq, nbshots=0, modulation=True)


# Simulate the Job with IsingAQPU using pulser-simulation
def _deploy_qpu(port):
    """Deploys a FresnelQPU using pulser-simulation on a local port."""
    np.random.seed(123)
    FresnelQPU(None).serve(port, "localhost")


port = 1234
server_thread = Thread(target=_deploy_qpu, args=(port,), daemon=True)
server_thread.start()

backends = {
    "pulser-simulation": None,
    "FresnelQPU": FresnelQPU(None),  # uses pulser-simulation
    "Remote FresnelQPU": RemoteQPU(port, "localhost"),
}
for msg, qpu in backends.items():
    aqpu = IsingAQPU.from_sequence(seq, qpu=qpu)
    np.random.seed(123)
    myqlm_results = aqpu.submit(job)
    out_myqlm_results = {
        sample.state.__str__(): sample.probability for sample in myqlm_results
    }
    results = IsingAQPU.convert_result_to_samples(myqlm_results)
    print(
        f"MyQLM Result obtained using IsingAQPU with {msg} with "
        "default number of samples (2000):"
    )
    print(myqlm_results, "\n")
    print("Expressed as a dictionary of (state: probability): ")
    print(out_myqlm_results, "\n")
    print("Converted into a Pulser Result:")
    print(results, "\n")

    assert results == exp_res
    assert out_myqlm_results == exp_myqlm_res

# Simulate the Job using AnalogQPU
try:
    from qlmaas.qpus import AnalogQPU

    analog_qpu = AnalogQPU()
    aqpu = IsingAQPU.from_sequence(seq, qpu=analog_qpu)
    analog_results = aqpu.submit(job)
    # Display the results once they have run on AnalogQPU
    print("Results obtained with AnalogQPU: ")
    print(analog_results.join())
    out_analog_results = {
        sample.state.__str__(): sample.probability for sample in analog_results
    }
    print("Expressed as a dictionary of (state: probability): ")
    print(out_analog_results, "\n")

    assert out_analog_results == {
        "|000>": 0.999372931965947,
        "|001>": 2.031335870809863e-07,
        "|010>": 2.0313358708098247e-07,
        "|011>": 5.266724155671983e-12,
        "|100>": 2.0313358708098515e-07,
        "|101>": 5.2667241556720065e-12,
        "|110>": 5.266724155672103e-12,
    }
except ImportError:
    # Shows as warning
    print(
        "\033[93m"
        + "Can't import AnalogQPU, check connection to Qaptiva Access."
        + "\033[0m"
    )

# Shows in green
print("\033[92m" + "All the simulations and checks are completed" + "\033[0m")

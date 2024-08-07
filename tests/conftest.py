import numpy as np
import pytest
from pulser import Pulse, Sequence
from pulser.waveforms import CustomWaveform
from qat.core import Schedule
from qat.core.variables import Variable

from pulser_myqlm import IsingAQPU
from pulser_myqlm.fresnel_qpu import TEMP_DEVICE


@pytest.fixture
def test_ising_qpu() -> IsingAQPU:
    return IsingAQPU(
        TEMP_DEVICE,
        TEMP_DEVICE.pre_calibrated_layouts[0].define_register(26, 35, 30),
    )


@pytest.fixture
def t_variable():
    return Variable("t")  # in ns


@pytest.fixture
def u_variable():
    return Variable("u")  # parameter


@pytest.fixture
def omega_t(t_variable):
    return (t_variable + 1) / 100


@pytest.fixture
def delta_t(t_variable, u_variable):
    return (1 - t_variable + u_variable) / 100  # in rad/us


@pytest.fixture
def schedule_seq(test_ising_qpu, omega_t, delta_t) -> tuple[Schedule, Sequence]:
    """A tuple of (Schedule, Sequence) who are equivalent."""
    t0 = 16 / 1000  # in µs
    H0 = test_ising_qpu.hamiltonian(omega_t, delta_t, 0)
    t1 = 24 / 1000  # in µs
    H1 = test_ising_qpu.hamiltonian(1, 0, 0)
    t2 = 20 / 1000  # in µs
    H2 = test_ising_qpu.hamiltonian(1, 0, np.pi / 2)

    schedule0 = Schedule(drive=[(1, H0)], tmax=t0)
    schedule1 = Schedule(drive=[(1, H1)], tmax=t1)
    schedule2 = Schedule(drive=[(1, H2)], tmax=t2)
    schedule = schedule0 | schedule1 | schedule2

    # Which is equivalent to having defined pulses using a Sequence
    seq = Sequence(test_ising_qpu.register, test_ising_qpu.device)
    seq.declare_channel("ryd_glob", "rydberg_global")

    seq.add(
        Pulse(
            CustomWaveform([omega_t(t=ti / 1000) for ti in range(int(t0 * 1000))]),
            CustomWaveform(
                [delta_t(t=ti / 1000, u=0) for ti in range(int(t0 * 1000))]
            ),  # no parametrized sequence for the moment
            0,
        ),
        "ryd_glob",
    )
    seq.add(
        Pulse.ConstantPulse(int(t1 * 1000), 1, 0, 0), "ryd_glob", protocol="no-delay"
    )
    seq.add(
        Pulse.ConstantPulse(int(t2 * 1000), 1, 0, np.pi / 2),
        "ryd_glob",
        protocol="no-delay",
    )
    return (schedule, seq)

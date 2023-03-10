import pytest
from pulser.devices import MockDevice
from pulser.register import Register
from qat.core.variables import Variable

from pulser_myqlm import IsingAQPU, PulserAQPU


@pytest.fixture
def test_pulser_qpu() -> PulserAQPU:
    return PulserAQPU(MockDevice, Register.square(2, 4, None))


@pytest.fixture
def test_ising_qpu() -> IsingAQPU:
    return IsingAQPU(MockDevice, Register.square(2, 4, None))


@pytest.fixture
def t_variable():
    return Variable("t")  # in ns


@pytest.fixture
def u_variable():
    return Variable("u")  # parameter


@pytest.fixture
def omega_t(t_variable):
    return t_variable + 1


@pytest.fixture
def delta_t(t_variable, u_variable):
    return 1 - t_variable + u_variable  # in rad/us

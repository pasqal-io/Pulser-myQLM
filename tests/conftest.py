import pytest
from qat.core.variables import Variable

from pulser_myqlm import IsingAQPU
from pulser_myqlm.pulserAQPU import TEMP_DEVICE


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

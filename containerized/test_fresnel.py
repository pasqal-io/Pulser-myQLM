"""Test to be run in the docker image."""

from __future__ import annotations

import os
from threading import Thread
from time import sleep

import numpy as np
import pytest
from pulser import Pulse, Sequence
from pulser.waveforms import CustomWaveform
from qat.comm.exceptions.ttypes import QPUException
from qat.core import Job, Sample, Schedule
from qat.core.qpu import QPUHandler
from qat.lang.AQASM import CCNOT, Program
from qat.qpus import RemoteQPU
from thrift.transport.TTransport import TTransportException

from pulser_myqlm import IsingAQPU
from pulser_myqlm.fresnel_qpu import FresnelQPU, TEMP_DEVICE


def compare_results_raw_data(results1: list, results2: list[tuple]) -> None:
    """Check that two lists of samples (as Result.raw_data) are the same."""
    for i, sample1 in enumerate(results1):
        res_sample1 = (sample1.probability, sample1._state, sample1.state.__str__())
        res_sample2 = (
            results2[i][0].probability,
            results2[i][0]._state,
            results2[i][1],
        )
        assert res_sample1 == res_sample2


BASE_URI = os.environ.get("PASQOS_URI", None)
PORT = 1190



@pytest.fixture
def test_ising_qpu() -> IsingAQPU:
    return IsingAQPU(
        TEMP_DEVICE,
        TEMP_DEVICE.pre_calibrated_layouts[0].define_register(26, 35, 30),
    )


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


@pytest.fixture
def circuit_job():
    """Circuit job."""
    # IsingQPU and FresnelQPU can only run job with a schedule
    # Defining a job from a circuit instead of a schedule
    prog = Program()
    qbits = prog.qalloc(CCNOT.arity)
    CCNOT(qbits)
    return prog.to_circ().to_job()


def deploy_qpu(qpu: QPUHandler, port: int) -> None:
    """Deploys the QPU on a server on a port at IP 127.0.0.1."""
    qpu.serve(port, "127.0.0.1")


def get_remote_qpu(port: int) -> RemoteQPU:
    """Get remote qpu for port."""
    tries = 0
    while tries < 10:
        try:
            return RemoteQPU(port, "localhost")
        except TTransportException as e:
            tries += 1
            sleep(1)
            error = e
    raise error


@pytest.mark.skipif(BASE_URI, reason="Emulation is run in CI.")
def test_run_sequence_fresnel_emulated(
    schedule_seq: tuple[Schedule, Sequence], circuit_job
):
    """Test simulation of a Sequence using pulser-simulation."""
    np.random.seed(123)
    _, seq = schedule_seq
    # pulser-simulation in a Remote FresnelQPU is used
    # Deploying a FresnelQPU on a remote server using serve
    server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), PORT))
    server_thread.daemon = True
    server_thread.start()
    # Accessing it with RemoteQPU
    sim_qpu = get_remote_qpu(PORT)

    aqpu = IsingAQPU.from_sequence(seq, qpu=sim_qpu)
    # FresnelQPU can only run job with a schedule
    # Defining a job from a circuit instead of a schedule
    with pytest.raises(
        QPUException, match="FresnelQPU can only execute a schedule job."
    ):
        aqpu.submit(circuit_job)

    # Run job created from a sequence using convert_sequence_to_job
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq, nbshots=1000)
    assert job_from_seq.nbshots == 1000
    result = aqpu.submit(job_from_seq)
    exp_result = [
        (Sample(probability=0.999, state=0), "|000>"),
        (Sample(probability=0.001, state=4), "|100>"),
    ]
    compare_results_raw_data(result.raw_data, exp_result)
    assert IsingAQPU.convert_result_to_samples(result) == {"000": 999, "100": 1}
    # Run job created from a sequence using convert_sequence_to_schedule
    schedule_from_seq = aqpu.convert_sequence_to_schedule(seq)
    job_from_seq = schedule_from_seq.to_job()  # manually defining number of shots
    assert not job_from_seq.nbshots
    result_schedule = aqpu.submit(job_from_seq)
    exp_result_schedule = [
        (Sample(probability=0.9995, state=0), "|000>"),
        (Sample(probability=0.0005, state=1), "|001>"),
    ]
    compare_results_raw_data(result_schedule.raw_data, exp_result_schedule)
    assert IsingAQPU.convert_result_to_samples(result_schedule) == {
        "000": 1999,
        "001": 1,
    }

    # Can simulate Job if Schedule is not equivalent to Sequence
    empty_job = Job()
    empty_schedule = Schedule()
    empty_schedule._other = schedule_from_seq._other
    empty_job.schedule = empty_schedule
    result_empty_sch = aqpu.submit(empty_job)
    exp_result_empty_sch = [
        (Sample(probability=0.999, state=0), "|000>"),
        (Sample(probability=0.0005, state=1), "|001>"),
        (Sample(probability=0.0005, state=4), "|100>"),
    ]
    compare_results_raw_data(result_empty_sch.raw_data, exp_result_empty_sch)
    assert IsingAQPU.convert_result_to_samples(result_empty_sch) == {
        "000": 1998,
        "001": 1,
        "100": 1,
    }


@pytest.mark.skipif(not BASE_URI, reason="CI can only run emulation.")
def test_run_sequence_fresnel_pasqos(
    schedule_seq: tuple[Schedule, Sequence], circuit_job
):
    """Test simulation of a Sequence using pulser-simulation."""
    np.random.seed(123)
    _, seq = schedule_seq
    # pulser-simulation in a Remote FresnelQPU is used
    # Deploying a FresnelQPU on a remote server using serve
    server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(BASE_URI), PORT))
    server_thread.daemon = True
    server_thread.start()
    # Accessing it with RemoteQPU
    sim_qpu = get_remote_qpu(PORT)

    aqpu = IsingAQPU.from_sequence(seq, qpu=sim_qpu)

    # Run job created from a sequence using convert_sequence_to_job
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq, nbshots=1000)
    assert job_from_seq.nbshots == 1000
    result = aqpu.submit(job_from_seq)
    exp_result = [
        (Sample(probability=0.999, state=0), "|000>"),
        (Sample(probability=0.001, state=4), "|100>"),
    ]
    compare_results_raw_data(result.raw_data, exp_result)
    assert IsingAQPU.convert_result_to_samples(result) == {"000": 999, "100": 1}
    # Run job created from a sequence using convert_sequence_to_schedule
    schedule_from_seq = aqpu.convert_sequence_to_schedule(seq)
    job_from_seq = schedule_from_seq.to_job()  # manually defining number of shots
    assert not job_from_seq.nbshots
    result_schedule = aqpu.submit(job_from_seq)
    exp_result_schedule = [
        (Sample(probability=0.9995, state=0), "|000>"),
        (Sample(probability=0.0005, state=1), "|001>"),
    ]
    compare_results_raw_data(result_schedule.raw_data, exp_result_schedule)
    assert IsingAQPU.convert_result_to_samples(result_schedule) == {
        "000": 1999,
        "001": 1,
    }

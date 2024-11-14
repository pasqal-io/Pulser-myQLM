"""Test to be run in the docker image."""

from __future__ import annotations

import os
from threading import Thread

import numpy as np
import pytest
from pulser import Sequence
from qat.comm.exceptions.ttypes import QPUException
from qat.core import Job, Sample, Schedule

from pulser_myqlm import IsingAQPU
from pulser_myqlm.fresnel_qpu import FresnelQPU
from tests.helpers.compare_raw_data import compare_results_raw_data
from tests.helpers.deploy_qpu import deploy_qpu, get_remote_qpu

BASE_URI = os.environ.get("PASQOS_URI", None)
PORT = 1190


@pytest.mark.skipif(BASE_URI is not None, reason="Emulation is run in CI.")
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

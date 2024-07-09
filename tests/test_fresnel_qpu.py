from __future__ import annotations

import json
from threading import Thread
from time import sleep
from unittest import mock

import numpy as np
import pytest
from pulser import Pulse, Sequence
from pulser.devices import MockDevice
from pulser.register import Register
from qat.comm.exceptions.ttypes import QPUException
from qat.core import Job, Sample, Schedule
from qat.core.qpu import QPUHandler
from qat.lang.AQASM import CCNOT, Program
from qat.qpus import PyLinalg, RemoteQPU
from thrift.transport.TTransport import TTransportException

from pulser_myqlm.constants import TEMP_DEVICE
from pulser_myqlm.fresnel_qpu import FresnelQPU
from pulser_myqlm.pulserAQPU import IsingAQPU


def deploy_qpu(qpu: QPUHandler, port: int) -> None:
    """Deploys the QPU on a server on a port at IP 127.0.0.1."""
    qpu.serve(port, "localhost")

def get_remote_qpu(port: int) -> RemoteQPU:
    tries = 0
    while tries < 10:
        try:
            return RemoteQPU(port, "localhost")
        except TTransportException as e:
            tries += 1
            sleep(1)
            error = e
    raise error


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

@pytest.fixture
def port() -> int:
    return 1190

@pytest.mark.parametrize("qpu", [None, "fresnel", "remote"])
@pytest.mark.parametrize("qpu", [None, "local", "remote"])
def test_run_sequence(schedule_seq, qpu):
    """Test simulation of a Sequence using pulser-simulation."""
    np.random.seed(123)
    schedule, seq = schedule_seq
    sim_qpu = None
    # If qpu is None, pulser-simulation in IsingAQPU is used
    if qpu == "local":
        # pulser-simulation in FresnelQPU is used
        sim_qpu = FresnelQPU(None)
        assert sim_qpu.is_operational
        sim_qpu.check_system()
    if qpu == "remote":
        # pulser-simulation in a Remote FresnelQPU is used
        # Deploying a FresnelQPU on a remote server using serve
        server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), port))
        server_thread.daemon = True
        server_thread.start()
        # Accessing it with RemoteQPU
        sim_qpu = get_remote_qpu(port)

    aqpu = IsingAQPU.from_sequence(seq, qpu=sim_qpu)
    # IsingQPU and FresnelQPU can only run job with a schedule
    # Defining a job from a circuit instead of a schedule
    prog = Program()
    qbits = prog.qalloc(CCNOT.arity)
    CCNOT(qbits)
    job = prog.to_circ().to_job()
    with pytest.raises(
        QPUException, match="FresnelQPU can only execute a schedule job."
    ):
        aqpu.submit(job)

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

    # Submit_job of IsingAQPU must not be used if qpu is not None
    if qpu is not None:
        with pytest.raises(
            ValueError,
            match="`submit_job` must not be used if the qpu attribute is defined,",
        ):
            aqpu.submit_job(schedule.to_job())

    # Can't simulate a time-dependent job with PyLinalg (a simulator of circuits)
    aqpu.set_qpu(PyLinalg())
    with pytest.raises(TypeError, match="'NoneType' object is not"):
        aqpu.submit(schedule.to_job())


class MockResponse:
    def __init__(self, json_data, status_code):
        """An object similar to an output of requests.get or requests.post."""
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def text(self):
        return ""


def mocked_requests_get_success(*args, **kwargs):
    """Mocks a requests.get response from a working system with successful jobs."""
    operational_url = "http://fresneldevice/api/latest/system/operational"
    job_url = "http://fresneldevice/api/latest/jobs/1"
    if args[0] == operational_url if args else kwargs["url"] == operational_url:
        return MockResponse({"data": {"operational_status": "UP"}}, 200)
    elif args[0] == job_url if args else kwargs["url"] == job_url:
        return MockResponse(
            {
                "data": {
                    "status": "DONE",
                    "result": json.dumps({"counter": {"000": 0.999, "100": 0.001}}),
                }
            },
            200,
        )
    return MockResponse(None, 404)


def mocked_requests_get_non_operational(*args, **kwargs):
    """Mocks a requests.get response from a non-working system."""
    operational_url = "http://fresneldevice/api/latest/system/operational"
    job_url = "http://fresneldevice/api/latest/jobs/1"
    if args[0] == operational_url if args else kwargs["url"] == operational_url:
        return MockResponse({"data": {"operational_status": "DOWN"}}, 200)
    elif args[0] == job_url if args else kwargs["url"] == job_url:
        return MockResponse({"data": {"status": "ERROR"}}, 200)
    return MockResponse(None, 404)


def mocked_requests_get_error(*args, **kwargs):
    """Mocks a requests.get response from a working system with non-working jobs."""
    operational_url = "http://fresneldevice/api/latest/system/operational"
    job_url = "http://fresneldevice/api/latest/jobs/1"
    if args[0] == operational_url if args else kwargs["url"] == operational_url:
        return MockResponse({"data": {"operational_status": "UP"}}, 200)
    elif args[0] == job_url if args else kwargs["url"] == job_url:
        return MockResponse({"data": {"status": "ERROR"}}, 200)
    return MockResponse(None, 404)


def mocked_requests_post_success(*args, **kwargs):
    """Mocks a response to the post of a job accepted by the system."""
    job_url = "http://fresneldevice/api/latest/jobs"
    if args[0] == job_url if args else kwargs["url"] == job_url:
        if list(kwargs["json"].keys()) != ["nb_run", "pulser_sequence"]:
            return MockResponse(None, 400)
        return MockResponse({"data": {"status": "PENDING", "uid": 1}}, 200)
    return MockResponse(None, 404)


def mocked_requests_post_fail(*args, **kwargs):
    """Mocks a response to the post of a job not accepted by the system."""
    job_url = "http://fresneldevice/api/latest/jobs"
    if args[0] == job_url if args else kwargs["url"] == job_url:
        if set(kwargs["json"].keys()) != ["nb_run", "pulser_sequence"]:
            return MockResponse(None, 400)
        return MockResponse({"data": {"status": "ERROR", "uid": 1}}, 500)
    return MockResponse(None, 404)

base_uris = ["http://fresneldevice/api", None]


def _switch_seq_device(seq, device):
    if device != TEMP_DEVICE:
        if device == MockDevice:
            with pytest.warns(UserWarning, match="Switching to a device"):
                seq = seq.switch_device(device)
        else:
            seq = seq.switch_device(device)
    return seq


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_success
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post",
    side_effect=mocked_requests_post_success,
)
@pytest.mark.parametrize("base_uri", base_uris)
@pytest.mark.parametrize("remote_fresnel", [False, True])
def test_job_submission(mock_get, mock_post, base_uri, remote_fresnel, schedule_seq, port: int):
    """Test submission of Jobs to a FresnelQPU interfacing a working QPU."""
    # Can't connect with a wrong address
    with pytest.raises(QPUException, match="Connection with API failed"):
        FresnelQPU(base_uri="")

    fresnel_qpu = FresnelQPU(base_uri=base_uri)

    # Deploy the QPU on a Qaptiva server
    if remote_fresnel:
        port += 1
        server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, port))
        server_thread.daemon = True
        server_thread.start()

    # Can't submit if the Device of the Sequence does not match the TEMP_DEVICE
    _, seq = schedule_seq
    mock_seq = _switch_seq_device(seq, MockDevice)
    job_from_seq = IsingAQPU.convert_sequence_to_job(mock_seq)
    qpu = get_remote_qpu(port) if remote_fresnel else fresnel_qpu
    with pytest.raises(QPUException, match="The Sequence in job.schedule._other"):
        qpu.submit(job_from_seq)

    # Can't simulate if Register is not from calibrated Layouts
    seq = Sequence(Register.triangular_lattice(2, 2, spacing=5), TEMP_DEVICE)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(Pulse.ConstantPulse(100, 1.0, 0.0, 0.0), "rydberg_global")
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(QPUException, match="The Register of the Sequence"):
        qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.fresnel_qpu.requests.get", side_effect=mocked_requests_get_success
)
@mock.patch(
    "pulser_myqlm.fresnel_qpu.requests.post",
    side_effect=mocked_requests_post_success,
)
@pytest.mark.parametrize("base_uri", ["http://fresneldevice/api", None])
@pytest.mark.parametrize("device", [TEMP_DEVICE, TEMP_DEVICE.to_virtual()])
def test_job_simulation(mock_get, mock_post, base_uri, device, schedule_seq):
    np.random.seed(123)

    # Modify the device of the Sequence
    _, seq = schedule_seq
    seq = _switch_seq_device(seq, device)
    fresnel_qpu = FresnelQPU(base_uri=base_uri)

    # Simulate Sequence using Pulser Simulation
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    result = fresnel_qpu.submit(job_from_seq)
    exp_result = [
        (Sample(probability=0.999, state=0), "|000>"),
        (Sample(probability=0.001, state=4), "|100>"),
    ]
    compare_results_raw_data(result.raw_data, exp_result)


class SideEffect:
    """Helper class to iterate through functions when calling side_effect."""

    def __init__(self, *fns):
        self.fs = iter(fns)

    def __call__(self, *args, **kwargs):
        f = next(self.fs)
        return f(*args, **kwargs)


@mock.patch("pulser_myqlm.fresnel_qpu.requests.post")
@mock.patch("pulser_myqlm.fresnel_qpu.requests.get")
@mock.patch("pulser_myqlm.fresnel_qpu.QPU_POLLING_INTERVAL_SECONDS")
def test_non_operational_qpu(polling_interval, mock_get, mock_post, schedule_seq):
    """Test the impact of non operational QPU on the flow of submitting a job.

    At first the FresnelQPU is instantiated with a non operational
    QPU. This should not prevent the FresnelQPU from being instantiated.

    Test the poll_system method. At first with a non operational response
    which triggers a user warning. On second attempt with an operational
    QPU response which breaks out of the polling loop.

    Do the same for deploy_qpu which also uses the poll_system method.

    Finally do the same for submit_job which also uses the poll_system method.
    """
    # Decrease polling_interval to 0.1 to speed up test.
    polling_interval.side_effect = 0.1

    # Set response to non operational for
    # - FresnelQPU instantiation
    # - is_operational check
    mock_get.side_effect = mocked_requests_get_non_operational
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    assert not fresnel_qpu.is_operational

    # Set response to non operational for first polling atempt
    # Set response to success in second polling attempt
    mock_get.side_effect = SideEffect(
        mocked_requests_get_non_operational, mocked_requests_get_success
    )
    with pytest.warns(UserWarning, match="QPU not operational, will try again in"):
        fresnel_qpu.poll_system()

    # Deploy the QPU on a Qaptiva server
    server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, 1236))
    server_thread.daemon = True
    # Set response to non operational for first polling atempt
    # Set response to success in second polling attempt
    mock_get.side_effect = SideEffect(
        mocked_requests_get_non_operational, mocked_requests_get_success
    )
    with pytest.warns(UserWarning, match="QPU not operational, will try again in"):
        server_thread.start()

    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)

    # Set response to non operational for first polling atempt
    # Set response to success in second polling attempt
    # Set response to success for querying results
    mock_get.side_effect = SideEffect(
        mocked_requests_get_non_operational,
        mocked_requests_get_success,
        mocked_requests_get_success,
        mocked_requests_get_success,
    )
    # Set response to sucess for posting job
    mock_post.side_effect = mocked_requests_post_success
    with pytest.warns(UserWarning, match="QPU not operational, will try again in"):
        fresnel_qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_success
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post", side_effect=mocked_requests_post_fail
)
@pytest.mark.parametrize("remote_fresnel", [False, True])
def test_submission_error(mock_get, mock_post, remote_fresnel, schedule_seq, port: int):
    """Test a FresnelQPU interfacing a working QPU that fails at launching jobs."""
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    if remote_fresnel:
        port += 1
        server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, port))
        server_thread.daemon = True
        server_thread.start()
    qpu = get_remote_qpu(port) if remote_fresnel else fresnel_qpu
    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(QPUException, match="Could not create job"):
        qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_error
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post",
    side_effect=mocked_requests_post_success,
)
@pytest.mark.parametrize("remote_fresnel", [False, True])
def test_execution_error(mock_get, mock_post, remote_fresnel, schedule_seq):
    """Test a FresnelQPU interfacing a non-working QPU which could accept jobs."""
    global port
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    if remote_fresnel:
        port += 1
        server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, port))
        server_thread.daemon = True
        server_thread.start()
    qpu = get_remote_qpu(port) if remote_fresnel else fresnel_qpu
    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(QPUException, match="An error occured,"):
        qpu.submit(job_from_seq)

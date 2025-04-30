"""Tests for PulserQLMConnection."""

import time
import typing
from dataclasses import replace
from threading import Thread
from unittest import mock

import pulser
import pytest
from pulser.backend.remote import JobParams
from qat.comm.exceptions.ttypes import QPUException
from qat.comm.qlmaas.ttypes import JobInfo, JobStatus
from qat.core import Batch, BatchResult, Result
from qat.qlmaas import QLMaaSConnection
from qat.qlmaas.result import AsyncResult
from qat.qlmaas.wrappers import ServiceDescription
from qat.qpus import PyLinalg, RemoteQPU

from pulser_myqlm import FresnelQPU, IsingAQPU, PulserQLMConnection
from pulser_myqlm.ising_aqpu import QPUType
from tests.helpers.deploy_qpu import deploy_qpu

FRESNEL_PORT = 1234


class MockJobInfo(JobInfo):
    def __init__(self, id: str, status: JobStatus) -> None:
        self.id = id
        self.status = status


class MockAsyncResult(AsyncResult):
    def __init__(self, job_id: str, connection: QLMaaSConnection):
        self.job_id = job_id
        self.connection = connection

    def get_batch(self) -> Batch:
        return self.connection.batchs[self.job_id][0]

    def get_info(self) -> JobInfo:
        return MockJobInfo(self.job_id, self.get_status(False))

    def get_result(self) -> BatchResult:
        if self.connection.batchs[self.job_id][2] == JobStatus.DONE:
            return self.connection.batchs[self.job_id][1]
        else:
            raise Exception  # I don't know the exception

    def get_status(self, human_readable: bool = True) -> JobStatus | str:
        status = self.connection.batchs[self.job_id][2]
        if not human_readable:
            return status
        raise NotImplementedError(
            "Human readable status is not implemented in MockAsyncResult."
        )

    def join(self) -> BatchResult:
        while self.get_status() != JobStatus.DONE:
            time.wait(1)
        return self.get_result()


class RemoteFresnelQPU(RemoteQPU):
    """A RemoteQPU connecting to a remote Fresnel QPU."""

    def __init__(self):
        self.name = "RemoteFresnelQPU"
        super().__init__(port=FRESNEL_PORT, ip="localhost")


class WrongRemoteFresnelQPU(RemoteQPU):
    """A RemoteQPU connected to the wrong port of a server."""

    def __init__(self):
        self.name = "RemoteFresnelQPU"
        super().__init__(port=FRESNEL_PORT + 1, ip="localhost")


class MockQLMaaSConnection(QLMaaSConnection):
    """An object similar to a QLMaaSConnection.

    Keyword Arguments:
        wrong_remote_fresnel_qpu: If True, will expose a RemoteQPU
            pointing to a wrong server, making get_specs() fail.
        latency: an optional amount of time to wait before answering
            the call.
    """

    def __init__(
        self,
        hostname: str = None,
        port: int = None,
        authentication: str = None,
        certificate: str = None,
        key: str = None,
        check_host: bool = None,
        proxy_host: str = None,
        proxy_port: int = None,
        timeout: int = None,
        **kwargs: dict[str, typing.Any],
    ):
        self.hostname = hostname
        self.port = port
        self.authentication = authentication
        self.certificate = (certificate,)
        self.key = (key,)
        self.check_host = (check_host,)
        self.proxy_host = (proxy_host,)
        self.proxy_port = (proxy_port,)
        self.timeout = (timeout,)
        self.wrong_remote_fresnel_qpu = kwargs.get("wrong_remote_fresnel_qpu", False)
        self.latency = kwargs.get("latency", 0)
        self.available_qpus = {
            "IsingAQPU": IsingAQPU,  # Needs args to be init
            "PyLinalg": PyLinalg,  # Doesn't have a descr
            "RemoteFresnelQPU": (
                RemoteFresnelQPU
                if not self.wrong_remote_fresnel_qpu
                else WrongRemoteFresnelQPU
            ),  # Has a device in its descr
        }
        self.batchs: dict[str, list[Batch, BatchResult | None, JobStatus]] = {}

    def get_qpus(self) -> list[ServiceDescription]:
        return [ServiceDescription(qpu_name) for qpu_name in self.available_qpus]

    def get_qpu(self, name: str) -> QPUType:
        return self.available_qpus[name]

    def _get_next_job_id(self):
        return str(len(self.batchs))

    def get_job(self, job_id: str) -> MockAsyncResult:
        return MockAsyncResult(job_id, self)

    def add_job(self, job_id: str, batch: Batch) -> None:
        self.batchs[job_id] = [batch, None, JobStatus.WAITING]

    def run_job(self, job_id: str) -> None:
        self.batchs[job_id][2] = JobStatus.RUNNING

    def associate_result_to_job(self, job_id: str, result: BatchResult) -> None:
        self.batchs[job_id][1] = result
        self.batchs[job_id][2] = JobStatus.DONE


@mock.patch("pulser_myqlm.connection.qat.qlmaas.QLMaaSConnection", MockQLMaaSConnection)
def test_available_devices():
    server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), FRESNEL_PORT))
    server_thread.daemon = True
    server_thread.start()
    # Fetching available devices using this connection will fail,
    # because a connection error occured
    mock_conn = PulserQLMConnection(wrong_remote_fresnel_qpu=True)
    with pytest.raises(
        QPUException, match="TTransportException: Could not connect to any of "
    ):
        mock_conn.fetch_available_devices()
    # Only the QPUs that don't need arguments and have serialized device
    # in the description of their HardwareSpecs are shown.
    mock_conn = PulserQLMConnection()
    assert mock_conn.fetch_available_devices() == {
        "RemoteFresnelQPU": FresnelQPU(None).device
    }


@mock.patch("pulser_myqlm.connection.qat.qlmaas.QLMaaSConnection", MockQLMaaSConnection)
def test_seq_submission():
    global FRESNEL_PORT
    FRESNEL_PORT += 1
    server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), FRESNEL_PORT))
    server_thread.daemon = True
    server_thread.start()
    mock_conn = PulserQLMConnection()

    # To check job submission, QPU should return an AsyncResult
    class QLMaaSFresnelQPU(RemoteQPU):
        """A RemoteQPU connecting to a remote Fresnel QPU."""

        def __init__(self):
            self.name = "QLMaaSFresnelQPU"
            super().__init__(port=FRESNEL_PORT, ip="localhost")

        def submit(self, batch: Batch) -> AsyncResult:
            job_id = mock_conn._connection._get_next_job_id()
            mock_conn._connection.add_job(job_id, batch)
            mock_conn._connection.run_job(job_id)
            res = super().submit(batch)
            mock_conn._connection.associate_result_to_job(job_id, res)
            # async result needs connection
            return MockAsyncResult(job_id, mock_conn._connection)

    mock_conn._connection.available_qpus["RemoteFresnelQPU"] = QLMaaSFresnelQPU
    # Submitting a Sequence not built with a Device available on the Connection
    seq = pulser.Sequence(pulser.Register.square(2, 5), pulser.AnalogDevice)
    # No matter the Sequence, open batches are not supported
    with pytest.raises(NotImplementedError, match="Open batches are not implemented"):
        mock_conn.submit(seq, open=True)
    # Can't submit the Sequence since the measurement basis can't be determined
    with pytest.raises(
        ValueError, match="The measurement basis can't be implicitly determined"
    ):
        mock_conn.submit(seq)
    seq.measure("ground-rydberg")
    # Can't submit the Sequence since device name is not among available
    with pytest.raises(
        ValueError, match="The Sequence's device AnalogDevice doesn't match"
    ):
        mock_conn.submit(seq)
    # This is also raised if mimic_qpu is set to True
    with pytest.raises(
        ValueError, match="does not match any of the devices currently available"
    ):
        mock_conn.submit(seq, mimic_qpu=True)
    # Using a Device with the right name, but different specs
    seq = seq.switch_device(replace(pulser.MockDevice, name="Fresnel"))
    # Submitting without specifying the Job Parameters creates an empty batch
    res = mock_conn.submit(seq)
    assert isinstance(res, pulser.backend.remote.RemoteResults)
    assert res.batch_id == "0"
    assert res.job_ids == []
    assert res.get_batch_status() == pulser.backend.remote.BatchStatus.DONE
    assert res.get_available_results() == {}
    # Submitting to an already existing batch raises an error
    with pytest.raises(NotImplementedError, match="It is not possible to add jobs"):
        mock_conn.submit(seq, batch_id="0")
    # Submitting with Job Parameters
    with pytest.raises(AttributeError, match="'str' object has no attribute 'get'"):
        # must provide a list of JobParams TODO: Improve error message
        mock_conn.submit(seq, job_params=JobParams(runs=100, variables={}))
    # QPU doesn't accept a Sequence with a device with a different rydberg level
    assert seq.device.rydberg_level != (fresnel_device := FresnelQPU(None).device)
    with pytest.raises(QPUException, match="The Sequence in job.schedule._other"):
        mock_conn.submit(seq, job_params=[JobParams(runs=100, variables={})])
    # Using a Device with the right name and right Rydberg level
    seq = seq.switch_device(
        replace(seq.device, rydberg_level=fresnel_device.rydberg_level)
    )
    # QPU rejects the Sequence because it requires a layout
    assert fresnel_device.requires_layout
    with pytest.raises(QPUException, match="must be defined from a layout."):
        mock_conn.submit(seq, job_params=[JobParams(runs=100, variables={})])
    # QPU requires a layout defined from calibrated layouts
    assert not fresnel_device.accepts_new_layouts
    square_layout = pulser.register.special_layouts.SquareLatticeLayout(4, 4, 5)
    assert not fresnel_device.is_calibrated_layout(square_layout)
    seq._set_register(seq, square_layout.define_register(0, 1, 4, 5))
    with pytest.raises(
        QPUException, match="must be defined from one of the calibrated layouts"
    ):
        mock_conn.submit(seq, job_params=[JobParams(runs=100, variables={})])
    # Using a layout from calibrated layouts
    seq._set_register(
        seq,
        fresnel_device.pre_calibrated_layouts[0].define_register(
            21, 26, 35, 39, 34, 25
        ),
    )
    with pytest.raises(
        QPUException, match="The Sequence should contain at least one Pulse"
    ):
        res = mock_conn.submit(seq, job_params=[JobParams(runs=100, variables={})])

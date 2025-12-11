"""Tests for PulserQLMConnection."""

from __future__ import annotations

import logging
import time
import typing
from dataclasses import replace
from threading import Thread
from unittest import mock

import numpy as np
import pulser
import pytest
from pulser.backend.remote import JobParams
from qat.comm.exceptions.ttypes import QPUException
from qat.comm.qlmaas.ttypes import JobInfo, JobStatus
from qat.core import Batch, Job, Observable, Result, Schedule
from qat.qlmaas import QLMaaSConnection
from qat.qlmaas.result import AsyncResult
from qat.qlmaas.wrappers import ServiceDescription
from qat.qpus import PyLinalg, RemoteQPU

from pulser_myqlm import FresnelQPU, IsingAQPU, PulserQLMConnection
from pulser_myqlm.helpers.deserialize_other import deserialize_other
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

    def get_batch(self) -> Job:
        return self.connection.batchs[self.job_id][0]

    def get_info(self) -> JobInfo:
        return MockJobInfo(self.job_id, self.get_status())

    def get_result(self) -> Result:
        if self.connection.batchs[self.job_id][2] == JobStatus.DONE:
            return self.connection.batchs[self.job_id][1]
        else:
            raise Exception  # I don't know the exception

    def get_status(self, human_readable: bool = False) -> typing.Union[JobStatus, str]:
        status = self.connection.batchs[self.job_id][2]
        if not human_readable:
            return status
        raise NotImplementedError(
            "Human readable status is not implemented in MockAsyncResult."
        )

    def join(self) -> Result:
        while self.get_status() != JobStatus.DONE:
            time.wait(1)
        return self.get_result()

    def delete_files(self) -> None:
        """Tries to delete job files."""
        self.connection.batchs[self.job_id][2] = JobStatus.DELETED

    def cancel(self) -> None:
        """Tries to cancel the job."""
        self.connection.batchs[self.job_id][2] = JobStatus.CANCELLED


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
        hostname: str | None = None,
        port: int | None = None,
        authentication: str | None = None,
        certificate: str | None = None,
        key: str | None = None,
        check_host: bool | None = None,
        proxy_host: str | None = None,
        proxy_port: int | None = None,
        timeout: int | None = None,
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
        self.batchs: dict[str, list[Job, Result | None, JobStatus]] = {}

    def get_qpus(self) -> list[ServiceDescription]:
        return [ServiceDescription(qpu_name) for qpu_name in self.available_qpus]

    def get_qpu(self, name: str) -> QPUType:
        return self.available_qpus[name]

    def _get_next_job_id(self):
        return str(len(self.batchs))

    def get_job(self, job_id: str) -> MockAsyncResult:
        return MockAsyncResult(job_id, self)

    def add_job(self, job_id: str, job: Job) -> None:
        self.batchs[job_id] = [job, None, JobStatus.WAITING]

    def run_job(self, job_id: str) -> None:
        self.batchs[job_id][2] = JobStatus.RUNNING

    def associate_result_to_job(self, job_id: str, result: Result) -> None:
        self.batchs[job_id][1] = result
        self.batchs[job_id][2] = JobStatus.DONE

    def set_job_as_error(self, job_id: str) -> None:
        self.batchs[job_id][2] = JobStatus.FAILED


@mock.patch("pulser_myqlm.connection.qat.qlmaas.QLMaaSConnection", MockQLMaaSConnection)
def test_available_devices(caplog):
    server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), FRESNEL_PORT))
    server_thread.daemon = True
    server_thread.start()
    # If there is no QPU providing a pulser Device, returns no devices
    mock_conn = PulserQLMConnection(wrong_remote_fresnel_qpu=True)
    with caplog.at_level(logging.DEBUG):
        assert mock_conn.fetch_available_devices() == {}  # no devices
    # Get the reason of the failure in logs
    assert (
        "Can't find a correct Device in description of specs of QLMaaSQPU RemoteFresnelQPU. Got QPUException("
        in caplog.text
    )
    # Only the QPUs that don't need arguments and have serialized device
    # in the description of their HardwareSpecs are shown.
    mock_conn = PulserQLMConnection()
    assert mock_conn.fetch_available_devices() == {
        "RemoteFresnelQPU": FresnelQPU(None)._device
    }


@mock.patch("pulser_myqlm.connection.qat.qlmaas.QLMaaSConnection", MockQLMaaSConnection)
def test_seq_submission():
    global FRESNEL_PORT
    FRESNEL_PORT += 1
    server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), FRESNEL_PORT))
    server_thread.daemon = True
    server_thread.start()
    mock_conn = PulserQLMConnection()
    assert not mock_conn.supports_open_batch()

    # To check job submission, QPU should return an AsyncResult
    class QLMaaSFresnelQPU(RemoteQPU):
        """A RemoteQPU connecting to a remote Fresnel QPU."""

        def __init__(self):
            self.name = "QLMaaSFresnelQPU"
            super().__init__(port=FRESNEL_PORT, ip="localhost")

        def submit(self, batch: Job, raise_err: bool = True) -> AsyncResult:
            job_id = mock_conn.qlmaas_connection._get_next_job_id()
            mock_conn.qlmaas_connection.add_job(job_id, batch)
            mock_conn.qlmaas_connection.run_job(job_id)
            try:
                res = super().submit(batch)
            except Exception as e:
                mock_conn.qlmaas_connection.set_job_as_error(job_id)
                if raise_err:
                    raise e
                return MockAsyncResult(job_id, mock_conn.qlmaas_connection)
            mock_conn.qlmaas_connection.associate_result_to_job(job_id, res)
            # async result needs connection
            return MockAsyncResult(job_id, mock_conn.qlmaas_connection)

    mock_conn.qlmaas_connection.available_qpus["RemoteFresnelQPU"] = QLMaaSFresnelQPU
    # Submitting a Sequence not built with a Device available on the Connection
    seq = pulser.Sequence(pulser.Register.square(2, 5, prefix="q"), pulser.AnalogDevice)
    # [Open Batches] No matter the Sequence, open batches are not supported
    with pytest.raises(NotImplementedError, match="Open batches are not implemented"):
        mock_conn.submit(seq, open=True)
    # [Meas] Can't submit the Sequence since the measurement basis can't be determined
    with pytest.raises(
        ValueError, match="The measurement basis can't be implicitly determined"
    ):
        mock_conn.submit(seq)
    seq.declare_channel("rydberg_global", "rydberg_global")
    # [Device] Can't submit the Sequence since device name is not among available
    job_params = JobParams(runs=100, variables={})
    with pytest.raises(
        ValueError, match="The Sequence's device AnalogDevice doesn't match"
    ):
        mock_conn.submit(seq, job_params=[job_params])
    # [Device] This is also raised if mimic_qpu is set to True
    with pytest.raises(
        ValueError, match="does not match any of the devices currently available"
    ):
        mock_conn.submit(seq, job_params=[job_params], mimic_qpu=True)
    # Using a Device with the right name, but different specs
    seq = seq.switch_device(replace(pulser.MockDevice, name="Fresnel"))
    # [JobParams] Can't submit without specifying the Job Parameters
    with pytest.raises(ValueError, match="'job_params' must be specified"):
        mock_conn.submit(seq)
    # [JobParams] Must be a list of JobParams
    with pytest.raises(TypeError, match="'job_params' must be a list;"):
        mock_conn.submit(seq, job_params=JobParams(runs=100, variables={}))
    with pytest.raises(
        TypeError, match="All elements of 'job_params' must be dictionaries; "
    ):
        mock_conn.submit(seq, job_params=[100])
    # [Device] QPU doesn't accept Sequence with a device with a different rydberg level
    assert seq.device.rydberg_level != (fresnel_device := FresnelQPU(None)._device)
    with pytest.raises(QPUException, match="The Sequence in job.schedule._other"):
        mock_conn.submit(seq, job_params=[job_params])
    mock_conn._query_job_progress("0") == {"0": [pulser.backend.remote.JobStatus.ERROR]}
    with pytest.raises(ValueError, match="Job id 1 is invalid for batch"):
        mock_conn._fetch_result("0", ["1"])
    with pytest.raises(
        pulser.backend.remote.RemoteResultsError,
        match="The results are not yet available",
    ):
        mock_conn._fetch_result("0", ["0"])
    # [Device] This can be verified prior to submission with mimic_qpu
    with pytest.raises(
        ValueError, match="The sequence is not compatible with the latest device specs"
    ):
        mock_conn.submit(seq, job_params=[job_params], mimic_qpu=True)
    # [Device] Using a Device with the right name and right Rydberg level
    seq = seq.switch_device(replace(pulser.AnalogDevice, name="Fresnel"))
    # [JobParams] Don't accept job if too many runs in job params
    with pytest.raises(
        ValueError, match="All 'runs' must be below the maximum allowed by the device"
    ):
        mock_conn.submit(
            seq, job_params=[JobParams(runs=fresnel_device.max_runs + 1, variables={})]
        )
    # [Sequence] Checked prior to submission: Sequences can't be empty
    with pytest.raises(ValueError, match="'sequence' should not be empty"):
        mock_conn.submit(seq, job_params=[job_params], mimic_qpu=True)
    # [Register] QPU rejects the Sequence because it requires a layout
    assert fresnel_device.requires_layout
    with pytest.raises(QPUException, match="must be defined from a layout."):
        mock_conn.submit(seq, job_params=[job_params])
    # [Register] QPU requires a layout defined from calibrated layouts
    assert not fresnel_device.accepts_new_layouts
    square_layout = pulser.register.special_layouts.SquareLatticeLayout(4, 4, 5)
    assert not fresnel_device.is_calibrated_layout(square_layout)
    seq._set_register(seq, square_layout.define_register(0, 1, 4, 5))
    with pytest.raises(
        QPUException, match="must be defined from one of the calibrated layouts"
    ):
        mock_conn.submit(seq, job_params=[job_params])
    # [Register] Using a layout from calibrated layouts
    seq._set_register(
        seq,
        fresnel_device.pre_calibrated_layouts[0].define_register(6, 9, 54, 51),
    )
    assert seq.register.layout in fresnel_device.pre_calibrated_layouts
    # [Sequence] An empty sequence cannot run on the QPU
    with pytest.raises(
        QPUException, match="The Sequence should contain at least one Pulse"
    ):
        res = mock_conn.submit(seq, job_params=[job_params])
    # Adding a pulse to the Sequence
    seq.add(pulser.Pulse.ConstantPulse(1000, np.pi, 0, 0), "rydberg_global")
    # [Register] Checked prior to submission: Sequence's register requires a layout
    seq._set_register(seq, pulser.Register.square(2, 5, prefix="q"))
    assert fresnel_device.requires_layout
    with pytest.raises(
        ValueError,
        match="requires the sequence's register to be defined from a `RegisterLayout`",
    ):
        mock_conn.submit(seq, job_params=[job_params], mimic_qpu=True)
    # [Register] Checked prior to submission: Layout must be among calibrated layouts
    assert not fresnel_device.accepts_new_layouts
    seq._set_register(seq, square_layout.define_register(0, 1, 4, 5))
    # [Register] also checked prior to submission to the QPU
    with pytest.raises(
        ValueError, match="the register's layout must be one of the layouts available"
    ):
        mock_conn.submit(seq, job_params=[job_params], mimic_qpu=True)
    # [Register] Using a layout from calibrated layouts
    seq._set_register(
        seq,
        fresnel_device.pre_calibrated_layouts[0].define_register(6, 9, 54, 51),
    )
    # [JobParams] The max number of submission is also checked with mimic_qpu
    assert seq.register.layout in fresnel_device.pre_calibrated_layouts
    with pytest.raises(
        ValueError, match="All 'runs' must be below the maximum allowed by the device"
    ):
        mock_conn.submit(
            seq,
            job_params=[JobParams(runs=fresnel_device.max_runs + 1, variables={})],
            mimic_qpu=True,
        )
    job_id = mock_conn.qlmaas_connection._get_next_job_id()
    np.random.seed(111)
    res = mock_conn.submit(seq, job_params=[job_params])
    assert res.batch_id == f"{job_id}"
    assert res.job_ids == [f"{job_id}"]
    assert res.get_batch_status() == pulser.backend.remote.BatchStatus.DONE
    assert len(res.get_available_results()) == 1
    assert isinstance(
        res.get_available_results()[f"{job_id}"], pulser.result.SampledResult
    )
    assert res.get_available_results()[f"{job_id}"].bitstring_counts == {
        "1111": job_params["runs"]
    }
    assert res.get_available_results()[f"{job_id}"].atom_order == seq.register.qubit_ids
    assert res.get_available_results()[f"{job_id}"].meas_basis == "ground-rydberg"
    assert [sampled_res.bitstring_counts for sampled_res in res.results] == [
        {"1111": job_params["runs"]}
    ]
    assert mock_conn._fetch_result(f"{job_id}", None) == res.results
    # Submitting to an already existing batch raises an error
    with pytest.raises(NotImplementedError, match="It is not possible to add jobs"):
        mock_conn.submit(seq, batch_id=f"{job_id}")
        res = mock_conn.submit(seq, job_params=[job_params])
    # Working with a parametrized sequence
    var = seq.declare_variable("amp", dtype=float)
    seq.add(pulser.Pulse.ConstantPulse(1000, var, 0, 0), "rydberg_global")
    with pytest.raises(TypeError, match="Did not receive values for variables: amp"):
        mock_conn.submit(seq, job_params=[job_params])
    job_params["variables"] = {"nonamp": 2000, "amp": np.pi}
    with pytest.warns(UserWarning, match="No declared variables named: nonamp"):
        mock_conn.submit(seq, job_params=[job_params])
    # Submitting two jobs with the variable correctly defined
    job_id = int(mock_conn.qlmaas_connection._get_next_job_id())
    job_params["variables"] = {"amp": np.pi}
    job_params_2 = JobParams(runs=200, variables={"amp": np.pi})
    np.random.seed(111)
    res = mock_conn.submit(seq, job_params=[job_params, job_params_2], wait=True)
    assert res.batch_id == f"{job_id}|{job_id+1}"
    assert res.job_ids == [f"{job_id}", f"{job_id+1}"]
    assert res.get_batch_status() == pulser.backend.remote.BatchStatus.DONE
    assert len(res.get_available_results()) == 2
    assert res.get_available_results()[f"{job_id}"].bitstring_counts == {
        "0000": job_params["runs"]
    }
    assert res.get_available_results()[f"{job_id+1}"].bitstring_counts == {
        "0000": job_params_2["runs"]
    }
    seq.measure()
    assert (
        mock_conn.get_sequence(f"{job_id}").to_abstract_repr()
        == seq.build(amp=job_params["variables"]["amp"]).to_abstract_repr()
    )


@mock.patch("pulser_myqlm.connection.qat.qlmaas.QLMaaSConnection", MockQLMaaSConnection)
def test_result_fetching(circuit_job, schedule_seq):
    global FRESNEL_PORT
    FRESNEL_PORT += 1
    server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), FRESNEL_PORT))
    server_thread.daemon = True
    server_thread.start()
    mock_conn = PulserQLMConnection()
    # Can't fetch result of a circuit job
    mock_conn.qlmaas_connection.add_job("0", circuit_job)  # Job ID 0 is a circuit
    mock_conn.qlmaas_connection.batchs["0"][2] = JobStatus.DONE  # Assume it ran

    with pytest.raises(
        pulser.backend.remote.RemoteResultsError,
        match="The Job 0 does not have a schedule",
    ):
        mock_conn._query_job_progress("0")
    # Can't fetch result of a Job without a sequence
    mock_conn.qlmaas_connection.add_job("1", Schedule(Observable(2)).to_job())
    mock_conn.qlmaas_connection.batchs["1"][2] = JobStatus.DONE  # Assume it ran
    with pytest.raises(
        pulser.backend.remote.RemoteResultsError,
        match="Failed at finding a Sequence in the schedule of Job 1.",
    ):
        mock_conn._query_job_progress("1")
    # Can't fetch result of a Batch
    _, seq = schedule_seq
    job = IsingAQPU.convert_sequence_to_job(seq)
    mock_conn.qlmaas_connection.add_job("2", Batch([job, job]))
    mock_conn.qlmaas_connection.batchs["2"][2] = JobStatus.DONE  # Assume it ran

    with pytest.raises(
        pulser.backend.remote.RemoteResultsError,
        match="The Job 2 isn't a Job or a Batch with a single Job.",
    ):
        mock_conn._query_job_progress("2")
    # Can fetch result of a Batch with a single Job
    _, seq = schedule_seq
    job = IsingAQPU.convert_sequence_to_job(seq)
    mock_conn.qlmaas_connection.add_job("3", Batch([job]))
    mock_conn.qlmaas_connection.batchs["3"][2] = JobStatus.DONE  # Assume it ran
    assert mock_conn.get_sequence("3").to_abstract_repr() == seq.to_abstract_repr()


@mock.patch("pulser_myqlm.connection.qat.qlmaas.QLMaaSConnection", MockQLMaaSConnection)
def test_batch_status():
    global FRESNEL_PORT
    FRESNEL_PORT += 1
    server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), FRESNEL_PORT))
    server_thread.daemon = True
    server_thread.start()
    mock_conn = PulserQLMConnection()
    seq = pulser.Sequence(
        FresnelQPU(None)
        ._device.pre_calibrated_layouts[0]
        .define_register(6, 9, 54, 51),
        pulser.AnalogDevice,
    )
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(1000, np.pi, 0, 0), "rydberg_global")
    seq.measure("ground-rydberg")
    job1 = IsingAQPU.convert_sequence_to_job(seq, nbshots=100)
    job2 = IsingAQPU.convert_sequence_to_job(seq, nbshots=200)
    aqpu = IsingAQPU.from_sequence(seq)
    res1 = aqpu.submit(job1)
    res2 = aqpu.submit(job2)
    # Submit a batch 0|1
    mock_conn.qlmaas_connection.add_job("0", job1)
    mock_conn.qlmaas_connection.add_job("1", job2)
    mock_conn.qlmaas_connection.batchs["0"][1] = res1
    mock_conn.qlmaas_connection.batchs["1"][1] = res2
    assert [
        deserialize_other(job.schedule._other)["seq"].to_abstract_repr()
        for job in mock_conn.get_batch("0|1")
    ] == [seq.to_abstract_repr(), seq.to_abstract_repr()]
    # No job is running + all waiting -> they are pending
    assert mock_conn._get_batch_status("0") == pulser.backend.remote.BatchStatus.PENDING
    assert mock_conn._get_batch_status("1") == pulser.backend.remote.BatchStatus.PENDING
    assert (
        mock_conn._get_batch_status("0|1") == pulser.backend.remote.BatchStatus.PENDING
    )
    # If one is running -> batch is running
    mock_conn.qlmaas_connection.batchs["0"][2] = JobStatus.RUNNING
    assert (
        mock_conn._get_batch_status("0|1") == pulser.backend.remote.BatchStatus.RUNNING
    )
    # If one is done -> Pending until the other works
    mock_conn.qlmaas_connection.batchs["0"][2] = JobStatus.DONE
    assert (
        mock_conn._get_batch_status("0|1") == pulser.backend.remote.BatchStatus.PENDING
    )
    # If one done + one stopped -> Batch is PAUSED
    mock_conn.qlmaas_connection.batchs["1"][2] = JobStatus.STOPPED
    assert (
        mock_conn._get_batch_status("0|1") == pulser.backend.remote.BatchStatus.PAUSED
    )
    # If one done + one failed -> Batch is in error
    mock_conn.qlmaas_connection.batchs["1"][2] = JobStatus.FAILED
    assert mock_conn._get_batch_status("0|1") == pulser.backend.remote.BatchStatus.ERROR
    # Cancelling the batch
    mock_conn.cancel_batch("0|1")
    assert (
        mock_conn._get_batch_status("0|1") == pulser.backend.remote.BatchStatus.CANCELED
    )
    # Batch can be deleted
    mock_conn.qlmaas_connection.batchs["0"][2] = JobStatus.DONE
    mock_conn.qlmaas_connection.batchs["1"][2] = JobStatus.DONE
    mock_conn.delete_batch("0|1")
    assert (
        mock_conn._get_batch_status("0|1") == pulser.backend.remote.BatchStatus.CANCELED
    )
    # Unknown status raises an error
    mock_conn.qlmaas_connection.batchs["1"][2] = "UNKNOWN_STATUS"
    with pytest.raises(ValueError, match="Unknown Job status UNKNOWN_STATUS."):
        mock_conn._get_batch_status("0|1")

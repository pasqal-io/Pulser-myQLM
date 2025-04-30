"""Defines the PulserMyQLMBackend."""

from __future__ import annotations

import copy
import logging
import typing

import pulser
import qat
from pulser.backend.remote import BatchStatus, JobStatus
from pulser.exceptions.serialization import DeserializeDeviceError

from pulser_myqlm.helpers.deserialize_other import deserialize_other
from pulser_myqlm.helpers.qlm_connection import QLMServer
from pulser_myqlm.ising_aqpu import IsingAQPU

LOGGER = logging.getLogger(__name__)

JOB_STATUS_QLM_TO_PULSER_BATCH: dict[qat.comm.qlmaas.ttypes.JobStatus, BatchStatus] = {
    qat.comm.qlmaas.ttypes.JobStatus.WAITING: BatchStatus.PENDING,
    qat.comm.qlmaas.ttypes.JobStatus.RUNNING: BatchStatus.RUNNING,
    qat.comm.qlmaas.ttypes.JobStatus.DONE: BatchStatus.DONE,
    qat.comm.qlmaas.ttypes.JobStatus.CANCELLED: BatchStatus.CANCELED,
    qat.comm.qlmaas.ttypes.JobStatus.UNKNOWN_JOB: BatchStatus.ERROR,
    qat.comm.qlmaas.ttypes.JobStatus.IN_BUCKET: BatchStatus.PENDING,
    qat.comm.qlmaas.ttypes.JobStatus.DELETED: BatchStatus.ERROR,
    qat.comm.qlmaas.ttypes.JobStatus.STOPPED: BatchStatus.CANCELED,
    qat.comm.qlmaas.ttypes.JobStatus.FAILED: BatchStatus.ERROR,
}

JOB_STATUS_QLM_TO_PULSER_JOB: dict[qat.comm.qlmaas.ttypes.JobStatus, JobStatus] = {
    qat.comm.qlmaas.ttypes.JobStatus.WAITING: JobStatus.PENDING,
    qat.comm.qlmaas.ttypes.JobStatus.RUNNING: JobStatus.RUNNING,
    qat.comm.qlmaas.ttypes.JobStatus.DONE: JobStatus.DONE,
    qat.comm.qlmaas.ttypes.JobStatus.CANCELLED: JobStatus.CANCELED,
    qat.comm.qlmaas.ttypes.JobStatus.UNKNOWN_JOB: JobStatus.ERROR,
    qat.comm.qlmaas.ttypes.JobStatus.IN_BUCKET: JobStatus.PENDING,
    qat.comm.qlmaas.ttypes.JobStatus.DELETED: JobStatus.ERROR,
    qat.comm.qlmaas.ttypes.JobStatus.STOPPED: JobStatus.CANCELED,
    qat.comm.qlmaas.ttypes.JobStatus.FAILED: JobStatus.ERROR,
}


class PulserQLMConnection(pulser.backend.remote.RemoteConnection):
    """A connection to a QLM, to submit Sequences to QPUs.

    Wraps qat.qlmaas.QLMaaSConnection to:
        - connect to a QLM.
        - Find the QPUs connected to this QLM that accept Sequences.
        - Find the pulser Device implemented by this QPU at the time
            of the connection.
        - Submit pulser Sequences to the QPU, for execution.
        - Retrieve the Results of the execution as pulser Results.

    Arguments:
        hostname: Hostname of the server.
        port: port listened by the server.
        authentication: Authentication method. The authentication is either
            "password" to login using a pair username/password or "ssl" to
            login using a SSL certificate.
        certificate: path to SSL certificate.
        key: path to SSL key.
        check_host: checks the certificate of the server.
        proxy_host: hostname of the HTTPS proxy (if you want to use an HTTPS proxy -
            using an HTTPS proxy is not compatible with SSL authentication)
        proxy_port: port of the HTTPS proxy (if you want to use an HTTPS proxy -
            using an HTTPS proxy is not compatible with SSL authentication)
        timeout: keep alive connection timeout.

    Keyword Arguments:
        All the other parameters to initialize a qat.qlmaas.QLMaaSConnection.

    Attribute:
        _connection: The QLMaaSConnection instantiated with the keyword
            arguments.
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
    ) -> None:
        self._connection = qat.qlmaas.QLMaaSConnection(
            hostname,
            port,
            authentication,
            certificate,
            key,
            check_host,
            proxy_host,
            proxy_port,
            timeout,
            **kwargs,
        )

    def supports_open_batch(self) -> bool:
        """Flag to confirm this class doesn't support open batch creation."""
        return False

    @staticmethod
    def _get_job_ids_from_batch(batch: qat.core.Batch) -> list[str]:
        """Generate the job IDs for a qat.core.Batch.

        Inside a Batch containing jobs Job_1, Job_2, .... Job_n, the job IDS
        are "1", "2", ..., "{n}".
        """
        return [f"{i}" for (i, _) in enumerate(batch.jobs)]

    @staticmethod
    def _convert_qlm_status_to_pulser_batch(
        job_status: qat.qlmaas.ttypes.JobStatus,
    ) -> BatchStatus:
        try:
            return JOB_STATUS_QLM_TO_PULSER_BATCH[job_status]
        except KeyError as e:
            raise RuntimeError(f"Unknown Job status {job_status}.") from e

    @staticmethod
    def _convert_qlm_status_to_pulser_job(
        job_status: qat.qlmaas.ttypes.JobStatus,
    ) -> JobStatus:
        try:
            return JOB_STATUS_QLM_TO_PULSER_JOB[job_status]
        except KeyError as e:
            raise ValueError(f"Unknown Job status {job_status}.") from e

    def submit(
        self,
        sequence: pulser.Sequence,
        wait: bool = False,
        open: bool = False,
        batch_id: str | None = None,
        **kwargs: typing.Any,
    ) -> pulser.backend.remote.RemoteResults:
        """Submits the sequence for execution on a remote Pasqal backend."""
        if open:
            raise NotImplementedError(
                "Open batches are not implemented in Qaptiva Access."
            )
        sequence = self._add_measurement_to_sequence(sequence)
        job_params: list[pulser.backend.remote.JobParams] = (
            pulser.json.utils.make_json_compatible(kwargs.get("job_params", []))
        )
        mimic_qpu: bool = kwargs.get("mimic_qpu", False)
        if mimic_qpu:
            sequence = self.update_sequence_device(sequence)
            pulser.QPUBackend.validate_job_params(job_params, sequence.device.max_runs)

        # In PasqalCloud, if batch_id is not empty, we can submit new jobs to a
        # batch we just created. This is not implemented in QLMaaSConnection.
        if batch_id:
            raise NotImplementedError(
                "It is not possible to add jobs to a previously created batch "
                "with Qaptiva Access."
            )
        # Create a new batch by submitting to the targeted qpu
        # Find the targeted QPU
        for qpu_id, device in self.fetch_available_devices().items():
            if sequence.device.name == device.name:
                break
        else:
            raise ValueError(
                f"The Sequence's device {sequence.device.name} doesn't match the "
                "name of a device of any available QPU. Select your device among"
                "fetch_available_devices() and change your Sequence's device using"
                "its switch_device method."
            )
        # Instantiate the targeted QPU
        connected_qpu = QLMServer.get_qpu(self._connection, qpu_id)
        # Create a batch of Jobs
        jobs = []
        # Each set of parameters create a different myQLM Job
        for params in job_params:
            seq_to_submit = copy.deepcopy(sequence)
            if sequence.is_parametrized() or sequence.is_register_mappable():
                vars = params.get("variables", {})
                seq_to_submit = sequence.build(**vars)
            assert not (
                seq_to_submit.is_parametrized() or seq_to_submit.is_register_mappable()
            )
            jobs.append(
                IsingAQPU.convert_sequence_to_job(
                    seq_to_submit,
                    nbshots=params.get("runs", 0),
                    modulation=False,
                )
            )
        batch = qat.core.Batch(jobs)  # Create a myQLM Batch
        async_res = connected_qpu.submit(batch)  # Submit it to the QPU
        if wait:
            # Returns the result of the job when it's done
            async_res.join()
        return pulser.backend.remote.RemoteResults(
            async_res.get_id(), self, PulserQLMConnection._get_job_ids_from_batch(batch)
        )

    def fetch_available_devices(self) -> dict[str, pulser.devices.Device]:
        """Fetches the devices available through this connection."""
        # Get all the myQLM QPUs available through the QLMaaSConnection
        qpus_names = QLMServer.get_qpus(self._connection)  # list of names of qpus
        devices = {}
        for qpu_name in qpus_names:
            # Instantiate the QPU
            # A myQLM QPU associated with a Pasqal QPU doesn't take any args
            try:
                qpu_server = QLMServer.get_qpu(self._connection, qpu_name)
            except (RuntimeError, TypeError):
                LOGGER.debug(
                    f"QLMaaSQPU {qpu_name} does not contain a Device (can't be "
                    "initialized without providing arguments)."
                )
                # Go to the next QPU if the instantiation failed
                continue
            # A myQLM QPU associated with a Pasqal QPU has a serialized device
            # in the description of its specs
            try:
                device = pulser.devices.Device.from_abstract_repr(
                    qpu_server.get_description()
                )
            except (TypeError, DeserializeDeviceError) as e:
                # Go to the next QPU if no device was found
                LOGGER.debug(
                    "Can't find a correct Device in description of specs of QLMaaSQPU "
                    f"{qpu_name}. Got {repr(e)}."
                )
                continue
            devices[qpu_name] = device
        return devices

    def _fetch_result(
        self, batch_id: str, job_ids: list[str] | None
    ) -> tuple[pulser.backend.Results, ...]:
        # The results are always sampled results
        jobs = self._query_job_progress(batch_id)

        if job_ids is None:
            job_ids = list(jobs.keys())

        results: list[pulser.backend.Results] = []
        for id in job_ids:
            if id not in jobs:
                raise ValueError(
                    f"Job id {id} is invalid, valid ids are {list(jobs.keys())}."
                )
            status, result = jobs[id]
            if status != JobStatus.DONE:
                raise pulser.backend.remote.RemoteResultsError(
                    f"The results are not yet available, job {id} status is {status}."
                )
            assert result is not None  # result is None if its status is not DONE
            results.append(result)

        return tuple(results)

    def _query_job_progress(
        self, batch_id: str
    ) -> typing.Mapping[str, tuple[JobStatus, pulser.backend.Results | None]]:
        """Fetches the status and results of all the jobs in a batch.

        Unlike `_fetch_result`, this method does not raise an error if some
        jobs in the batch do not have results.

        It returns a dictionary mapping the job ID to its status and results.
        """
        # In QLMaaSConnection, the Jobs inside a Batch have the status of the Batch
        async_res = QLMServer.get_job(self._connection, batch_id)
        qlm_batch = async_res.get_batch()
        qlm_status = async_res.get_status()
        # Link the status of the batch to a pulser JobStatus
        status = PulserQLMConnection._convert_qlm_status_to_pulser_job(qlm_status)
        job_ids = PulserQLMConnection._get_job_ids_from_batch(qlm_batch)
        if status != JobStatus.DONE:
            # If the Batch is not done, then all results are None
            return {job_id: (status, None) for job_id in job_ids}
        results: dict[str, tuple[JobStatus, pulser.result.Result | None]] = {}
        # If the Batch is DONE, fetch the myqlm Result associated to each of its Jobs
        for i, result in enumerate(async_res.get_result().results):
            # and make a pulser SampledResult, that needs the qubit ids and meas basis
            # that are in the submitted Sequence, that is in Job.schedule._other
            job = qlm_batch[i]  # fetch the myQLM Job
            job_id = job_ids[i]
            # Extract the Sequence in Job.schedule._other using deserialize_other
            if job.schedule is None:
                raise pulser.backend.remote.RemoteResultsError(
                    f"The Job {job_id} does not have a schedule: "
                    "it can't have run on the QPU.",
                )
            try:
                other_dict = deserialize_other(job.schedule._other)
            except ValueError as e:
                raise pulser.backend.remote.RemoteResultsError(
                    f"Failed at finding a Sequence in the schedule of Job {job_id}.",
                ) from e
            seq = other_dict["seq"]  # the submitted Sequence
            # Create a SmapledResult with qubit ids, meas basis of seq, result of Job
            results[job_id] = (
                status,
                pulser.result.SampledResult(
                    atom_order=seq.get_register(include_mappable=True).qubit_ids,
                    meas_basis=seq.get_measurement_basis(),
                    bitstring_counts=IsingAQPU.convert_result_to_samples(result),
                ),
            )
        return results

    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        """Gets the status of a batch from its ID."""
        qlm_status = QLMServer.get_job(self._connection, batch_id).get_status()
        return PulserQLMConnection._convert_qlm_status_to_pulser_batch(qlm_status)

    def _get_job_ids(self, batch_id: str) -> list[str]:
        """Gets all the job IDs within a batch."""
        qlm_batch = QLMServer.get_job(self._connection, batch_id).get_batch()
        return PulserQLMConnection._get_job_ids_from_batch(qlm_batch)

    def cancel_batch(self, batch_id: str) -> None:
        """Cancels a batch using its ID."""
        QLMServer.get_job(self._connection, batch_id).cancel()

    def delete_batch(self, batch_id: str) -> None:
        """Deletes the files of a batch using its ID."""
        QLMServer.get_job(self._connection, batch_id).delete_files()

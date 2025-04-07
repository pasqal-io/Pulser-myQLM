"""Defines the PulserMyQLMBackend."""

from __future__ import annotations

import time
import typing

import pulser
import qat
from pulser.backend.remote import BatchStatus, JobStatus
from pulser.json.exceptions import DeserializeDeviceError
from qat.comm.qlmaas.ttypes import QLMServiceException

from pulser_myqlm.helpers.deserialize_other import deserialize_other
from pulser_myqlm.ising_aqpu import IsingAQPU

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
        - Retrieve the Results of the execution as pulser results.

    Keyword Arguments:
        The arguments to initialize a qat.qlmaas.QLMaaSConnection.

    Attribute:
        _connection: The QLMaaSConnection instantiated with the keyword
            arguments.
    """

    def __init__(self, **kwargs: dict[str, typing.Any]) -> None:
        self._connection = qat.qlmaas.QLMaaSConnection(**kwargs)

    def supports_open_batch(self) -> bool:
        """Flag to confirm this class doesn't support open batch creation."""
        return False

    @staticmethod
    def _get_job_ids_from_batch(batch: qat.core.Batch) -> list[str]:
        """Generate the job IDs for a qat.core.Batch."""
        return [f"{i}" for (i, _) in enumerate(batch.jobs)]

    @staticmethod
    def _convert_qlm_status_to_pulser_batch(
        job_status: qat.qlmaas.ttypes.JobStatus,
    ) -> BatchStatus:
        try:
            return JOB_STATUS_QLM_TO_PULSER_BATCH[job_status]
        except IndexError as e:
            raise ValueError(f"Unknown Job status {job_status}.") from e

    @staticmethod
    def _convert_qlm_status_to_pulser_job(
        job_status: qat.qlmaas.ttypes.JobStatus,
    ) -> JobStatus:
        try:
            return JOB_STATUS_QLM_TO_PULSER_JOB[job_status]
        except IndexError as e:
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

        if sequence.is_parametrized() or sequence.is_register_mappable():

            for params in job_params:
                vars = params.get("variables", {})
                sequence.build(**vars)

        # In PasqalCloud, if batch_id is not empty, we can submit new jobs to a
        # batch we just created. This is not implemented in QLMaaSConnection.
        if batch_id:
            raise NotImplementedError(
                "It is not possible to add jobs to a previously created batch "
                "with Qaptiva Access."
            )
        # Create a new batch by submitting to the targetted qpu
        # Find the targeted QPU
        for qpu_id, device in self.fetch_available_devices().items():
            if sequence.device.name == device.name:
                break
        connected_qpu = self._connection.get_qpu(qpu_id)()
        jobs = []
        for params in job_params:
            if sequence.is_parametrized() or sequence.is_register_mappable():
                vars = params.get("variables", {})
                sequence = sequence.build(**vars)
            assert not (sequence.is_parametrized() or sequence.is_register_mappable())
            jobs.append(
                IsingAQPU.convert_sequence_to_job(
                    sequence,
                    nbshots=params.get("runs", 0),
                    modulation=False,
                )
            )
        batch = qat.core.Batch(jobs)
        async_res = connected_qpu.submit(batch)
        res_info = async_res.get_info()
        if wait:
            async_res.join()

        return pulser.backend.remote.RemoteResults(
            res_info.id, self, PulserQLMConnection._get_job_ids_from_batch(batch)
        )

    def fetch_available_devices(self) -> dict[str, pulser.devices.Device]:
        """Fetches the devices available through this connection."""
        qpus = self._connection.get_qpus()
        devices = {}
        for qpu in qpus:
            try:
                real_qpu = self._connection.get_qpu(qpu)()
            except RuntimeError:
                continue
            try:
                device = pulser.devices.Device.from_abstract_repr(
                    real_qpu.get_specs().description
                )
            except (TypeError, DeserializeDeviceError):
                continue
            devices[qpu.name] = device
        return devices

    def _fetch_result(
        self, batch_id: str, job_ids: list[str] | None
    ) -> tuple[pulser.result.Result, ...]:
        # The results are always sampled results
        jobs = self._query_job_progress(batch_id)

        if job_ids is None:
            job_ids = list(jobs.keys())

        results: list[pulser.result.Result] = []
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
    ) -> typing.Mapping[str, tuple[JobStatus, pulser.result.Result | None]]:
        """Fetches the status and results of all the jobs in a batch.

        Unlike `_fetch_result`, this method does not raise an error if some
        jobs in the batch do not have results.

        It returns a dictionary mapping the job ID to its status and results.
        """
        qlm_batch = self._get_batch(batch_id)
        qlm_status = self._connection.get_job(batch_id).get_status(human_readable=False)
        status = PulserQLMConnection._convert_qlm_status_to_pulser_job(qlm_status)
        job_ids = PulserQLMConnection._get_job_ids_from_batch(qlm_batch)
        if status != JobStatus.DONE:
            return {job_id: (status, None) for job_id in job_ids}
        results: dict[str, tuple[JobStatus, pulser.result.Result | None]] = {}
        for i, result in enumerate(
            self._connection.get_job(batch_id).get_result().results
        ):
            job = qlm_batch[i]
            job_id = job_ids[i]
            if job.schedule is None:
                raise pulser.backend.remote.RemoteResultsError(
                    f"The Job {job_id} does not have a schedule: "
                    "it can't have run on the QPU.",
                )
            try:
                other_dict = deserialize_other(job.schedule._other)
            except ValueError as e:
                raise pulser.backend.remote.RemoteResultsError(
                    "Failed at finding a Sequence in the schedule" f"of Job {job_id}.",
                ) from e
            seq = other_dict["seq"]
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
        qlm_status = self._connection.get_job(batch_id).get_status(human_readable=False)
        return PulserQLMConnection._convert_qlm_status_to_pulser_batch(qlm_status)

    def _get_batch(self, batch_id: str) -> qat.core.Batch:
        """Returns the myQLM Batch associated with a batch id."""
        qlm_batch = None
        for i in range(10):
            try:
                qlm_batch = self._connection.get_job(batch_id).get_batch()
                break
            except QLMServiceException:
                time.sleep(1)
        if qlm_batch is None:
            raise QLMServiceException(
                f"Max retries performed while trying to access batch of {batch_id}."
            )
        return qlm_batch

    def _get_job_ids(self, batch_id: str) -> list[str]:
        """Gets all the job IDs within a batch."""
        qlm_batch = self._get_batch(batch_id)
        return PulserQLMConnection._get_job_ids_from_batch(qlm_batch)

    def _close_batch(self, batch_id: str) -> None:
        """Closes a batch using its ID."""
        raise NotImplementedError(  # pragma: no cover
            "No batch can be open/closed through this remote connection."
            "It can be cancelled using `cancel_batch`."
        )

    def cancel_batch(self, batch_id: str) -> None:
        """Cancels a batch using its ID."""
        self._connection.get_job(batch_id).cancel()

    def delete_batch(self, batch_id: str) -> None:
        """Deletes the files of a batch using its ID."""
        self._connection.get_job(batch_id).delete_files()

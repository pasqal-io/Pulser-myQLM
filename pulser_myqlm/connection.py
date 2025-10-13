"""Defines the PulserMyQLMBackend."""

from __future__ import annotations

import logging
import typing

import pulser
import qat
from pulser.backend.remote import BatchStatus, JobStatus
from pulser.exceptions.serialization import DeserializeDeviceError

from pulser_myqlm.helpers.deserialize_other import deserialize_other
from pulser_myqlm.helpers.qlm_connection import QLMClient
from pulser_myqlm.ising_aqpu import IsingAQPU

logger = logging.getLogger(__name__)


JOB_STATUS_QLM_TO_PULSER_JOB: dict[qat.comm.qlmaas.ttypes.JobStatus, JobStatus] = {
    qat.comm.qlmaas.ttypes.JobStatus.WAITING: JobStatus.PENDING,
    qat.comm.qlmaas.ttypes.JobStatus.RUNNING: JobStatus.RUNNING,
    qat.comm.qlmaas.ttypes.JobStatus.DONE: JobStatus.DONE,
    qat.comm.qlmaas.ttypes.JobStatus.CANCELLED: JobStatus.CANCELED,
    qat.comm.qlmaas.ttypes.JobStatus.UNKNOWN_JOB: JobStatus.ERROR,
    qat.comm.qlmaas.ttypes.JobStatus.IN_BUCKET: JobStatus.PENDING,
    qat.comm.qlmaas.ttypes.JobStatus.DELETED: JobStatus.CANCELED,
    qat.comm.qlmaas.ttypes.JobStatus.STOPPED: JobStatus.PAUSED,
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
        kwargs.update(
            {
                "hostname": hostname,
                "port": port,
                "authentication": authentication,
                "certificate": certificate,
                "key": key,
                "check_host": check_host,
                "proxy_host": proxy_host,
                "proxy_port": proxy_port,
                "timeout": timeout,
            }
        )
        self._qlm_client = QLMClient(qat.qlmaas.QLMaaSConnection(**kwargs))

    @property
    def qlmaas_connection(self) -> qat.qlmaas.QLMaaSConnection:
        """The QLMaaSConnection used."""
        return self._qlm_client._connection

    def supports_open_batch(self) -> bool:
        """Flag to confirm this class doesn't support open batch creation."""
        return False

    @staticmethod
    def _batch_id_from_job_ids(job_ids: list[str]) -> str:
        """Generate the Batch ID from a list of Job IDs."""
        return "|".join(job_ids)

    @staticmethod
    def _get_job_ids(batch_id: str) -> list[str]:
        """Gets all the job IDs within a batch."""
        return batch_id.split("|")

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
        # Check that Job Params are correctly defined
        job_params: list[pulser.backend.remote.JobParams] = (
            pulser.json.utils.make_json_compatible(kwargs.get("job_params", []))
        )
        mimic_qpu: bool = kwargs.get("mimic_qpu", False)
        if mimic_qpu:
            # Replace the sequence's device by the QPU's
            sequence = self.update_sequence_device(sequence)
            # Check that the job params match with the max number of runs
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
        # Check JobParams
        pulser.QPUBackend.validate_job_params(job_params, device.max_runs)
        # Instantiate the targeted QPU
        connected_qpu = self._qlm_client.get_qpu(qpu_id)
        # Submit one myQLM Job per job params
        results = []
        for params in job_params:
            seq_to_submit = sequence
            if sequence.is_parametrized() or sequence.is_register_mappable():
                vars = params.get("variables", {})
                seq_to_submit = sequence.build(**vars)
            assert not (
                seq_to_submit.is_parametrized() or seq_to_submit.is_register_mappable()
            )
            results.append(
                # no backoff on submit to have Jobs submitted once
                connected_qpu.submit(
                    IsingAQPU.convert_sequence_to_job(
                        seq_to_submit,
                        nbshots=params.get("runs", 0),
                        modulation=False,
                    )
                )
            )
        if wait:
            for res in results:
                # Returns the result of the job when it's done
                res.join()
        job_ids = [res.get_id() for res in results]
        return pulser.backend.remote.RemoteResults(
            self._batch_id_from_job_ids(job_ids), self, job_ids
        )

    def fetch_available_devices(self) -> dict[str, pulser.devices.Device]:
        """Fetches the devices available through this connection."""
        # Get all the myQLM QPUs available through the QLMaaSConnection
        qpus_names = self._qlm_client.list_qpu_names()
        devices = {}
        for qpu_name in qpus_names:
            # Instantiate the QPU
            # A myQLM QPU associated with a Pasqal QPU doesn't take any args
            try:
                qpu = self._qlm_client.get_qpu(qpu_name)
            except Exception:
                logger.debug(
                    f"QLMaaSQPU {qpu_name} does not contain a Device (can't be "
                    "initialized without providing arguments)."
                )
                # Go to the next QPU if the instantiation failed
                continue
            # A myQLM QPU associated with a Pasqal QPU has a serialized device
            # in the description of its specs
            try:
                device = pulser.devices.Device.from_abstract_repr(qpu.get_description())
            except (TypeError, DeserializeDeviceError) as e:
                # Go to the next QPU if no device was found
                logger.debug(
                    "Can't find a correct Device in description of specs of QLMaaSQPU "
                    f"{qpu_name}. Got {repr(e)}."
                )
                continue
            devices[qpu_name] = device
        return devices

    def _fetch_result(
        self, batch_id: str, job_ids: list[str] | None
    ) -> tuple[pulser.backend.Results, ...]:
        """Fetches the results of a completed batch."""
        # The results are always sampled results
        jobs_progression = self._query_job_progress(batch_id)

        if job_ids is None:
            job_ids = list(jobs_progression.keys())

        results: list[pulser.backend.Results] = []
        for id in job_ids:
            if id not in jobs_progression:
                raise ValueError(
                    f"Job id {id} is invalid for batch {batch_id}, "
                    f"valid ids are {list(jobs_progression.keys())}."
                )
            status, result = jobs_progression[id]
            if status != JobStatus.DONE:
                raise pulser.backend.remote.RemoteResultsError(
                    f"The results are not yet available, job {id} has status {status}."
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
        job_ids = self._get_job_ids(batch_id)
        progress_results: dict[str, tuple[JobStatus, pulser.backend.Results | None]] = (
            {}
        )
        for job_id in job_ids:
            # Query the AsyncResult associated to each Job
            async_res = self._qlm_client.get_job(job_id)
            # Link the status of the Job to a pulser JobStatus
            qlm_status = async_res.get_status()
            status = self._convert_qlm_status_to_pulser_job(qlm_status)
            if status != JobStatus.DONE:
                # The result of a not DONE Job is None
                progress_results[job_id] = (status, None)
                continue
            # The Job is DONE, fetch its myqlm Result
            result = async_res.get_result()
            # and make a pulser SampledResult, that needs the qubit ids and meas basis
            # that are in the submitted Sequence.
            seq = self.get_sequence(job_id)
            # Create a SampledResult with qubit ids, meas basis of seq, result of Job
            progress_results[job_id] = (
                status,
                pulser.result.SampledResult(
                    atom_order=seq.get_register(include_mappable=True).qubit_ids,
                    meas_basis=seq.get_measurement_basis(),
                    bitstring_counts=IsingAQPU.convert_result_to_samples(result),
                ),
            )
        return progress_results

    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        """Gets the status of a batch from its ID."""
        jobs_progression = self._query_job_progress(batch_id)
        statuses = [
            progression_result[0]
            for (job, progression_result) in jobs_progression.items()
        ]
        if JobStatus.RUNNING in statuses:
            # Batch is RUNNING if at least one Job is running
            return BatchStatus.RUNNING
        # If no Job is RUNNING
        if JobStatus.PENDING in statuses:
            # Batch is PENDING if not all the Jobs have tried to run
            return BatchStatus.PENDING
        # If no Job is RUNNING/will be run anymore
        if JobStatus.PAUSED in statuses:
            # Batch is PAUSED if one Job is paused
            return BatchStatus.PAUSED
        if JobStatus.ERROR in statuses:
            # Batch is in ERROR if one Job is in error
            return BatchStatus.ERROR
        if JobStatus.CANCELED in statuses:
            # Batch was CANCELED if one of its Job was canceled
            return BatchStatus.CANCELED
        assert len(set(statuses)) == 1 and statuses[0] == JobStatus.DONE
        return BatchStatus.DONE

    def cancel_batch(self, batch_id: str) -> None:
        """Cancels a batch using its ID."""
        job_ids = self._get_job_ids(batch_id)
        for job_id in job_ids:
            self._qlm_client.get_job(job_id).cancel()

    def delete_batch(self, batch_id: str) -> None:
        """Deletes the files of a batch using its ID."""
        job_ids = self._get_job_ids(batch_id)
        for job_id in job_ids:
            self._qlm_client.get_job(job_id).delete_files()

    def get_batch(self, batch_id: str) -> qat.core.Batch:
        """Returns a Batch associated with a batch_id."""
        job_ids = self._get_job_ids(batch_id)
        jobs = []
        for job_id in job_ids:
            jobs.append(self._qlm_client.get_job(job_id).get_batch())
        return qat.core.Batch(jobs=jobs)

    def get_sequence(self, job_id: str) -> pulser.Sequence:
        """Returns the Sequence associated with a job_id in the QLMaaSConnection."""
        # Fetch the job associated with the ID
        qlm_job = self._qlm_client.get_job(job_id).get_batch()
        # Extract the Sequence in Job.schedule._other using deserialize_other
        # A Batch with a single Job can be casted to a Job
        if isinstance(qlm_job, qat.core.Batch) and len(qlm_job.jobs) == 1:
            qlm_job = qlm_job.jobs[0]
        # Can't work with something different from a Job
        if not isinstance(qlm_job, qat.core.Job):
            raise pulser.backend.remote.RemoteResultsError(
                f"The Job {job_id} isn't a Job or a Batch with a single Job."
            )
        # Can't work with a Job not having a Schedule
        if qlm_job.schedule is None:
            raise pulser.backend.remote.RemoteResultsError(
                f"The Job {job_id} does not have a schedule: "
                "it can't have run on the QPU.",
            )
        try:
            other_dict = deserialize_other(qlm_job.schedule._other)
        except ValueError as e:
            raise pulser.backend.remote.RemoteResultsError(
                f"Failed at finding a Sequence in the schedule of Job {job_id}.",
            ) from e
        return typing.cast(pulser.Sequence, other_dict["seq"])  # the submitted Sequence

"""Defines the PulserMyQLMBackend."""

import typing

import pulser
import qat
from pulser.json.exceptions import DeserializeDeviceError

from pulser_myqlm import IsingAQPU


class PulserQLMaaSConnection(pulser.backend.remote.RemoteConnection):
    """The connection to a MyQLM QPU."""

    def __init__(self, connection: qat.qlmaas.QLMaaSConnection):
        self._connection = connection

    def supports_open_batch(self) -> bool:
        """Flag to confirm this class doesn't support open batch creation."""
        return False

    def submit(
        self,
        sequence: pulser.Sequence,
        wait: bool = False,
        open: bool = False,
        batch_id: str | None = None,
        **kwargs: typing.Any,
    ) -> pulser.backend.remote.RemoteResult:
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
            sequence = self.update_sequence_device(sequence, job_params)
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
        connected_qpu = self._connection.get_qpu(qpu_id)
        jobs = []
        for params in job_params:
            if sequence.is_parametrized() or sequence.is_register_mappable():
                vars = params.get("variables", {})
                built_seq = sequence.build(**vars)
            else:
                built_seq = sequence.copy()
            assert not (built_seq.is_parametrized() or sequence.is_register_mappable())
            jobs.append(
                IsingAQPU.convert_sequence_to_job(
                    built_seq,
                    nbshots=params.get("runs", 0),
                    modulation=False,
                )
            )
        async_res = connected_qpu.submit(qat.core.Batch(jobs))
        res_info = async_res.get_info()
        if wait:
            async_res.join()
        return pulser.RemoteResults(res_info.id, self._connection)

    def fetch_available_devices(self) -> dict[str, pulser.devices.Device]:
        """Fetches the devices available through this connection."""
        qpus = self._connection.get_qpus()
        devices = {}
        for qpu in qpus:
            try:
                device = pulser.devices.Device.from_abstract_repr(
                    qpu.get_specs().description
                )
            except (TypeError, DeserializeDeviceError):
                continue
            devices[qpu] = device
        return devices

    def _fetch_result(
        self, batch_id: str, job_ids: list[str] | None
    ) -> typing.Sequence[qat.core.BatchResult | qat.core.Result]:
        """Fetches the results of a completed batch."""
        # Returns an error if job is not available
        return self._connection.get_job(batch_id).get_result()

    def _query_job_progress(
        self, batch_id: str
    ) -> tuple[
        qat.comm.qlmaas.ttypes.JobStatus, qat.core.BatchResult | qat.core.Result | None
    ]:
        """Fetches the status and results of all the jobs in a batch.

        Unlike `_fetch_result`, this method does not raise an error if some
        jobs in the batch do not have results.

        It returns a dictionary mapping the job ID to its status and results.
        """
        status = self.get_batch_status(batch_id)
        result = None
        if status.value == qat.comm.qlmaas.ttypes.JobStatus.DONE:
            result = self._connection.get_job(batch_id).get_result
        return (status, result)

    def _get_batch_status(self, batch_id: str) -> qat.comm.qlmaas.ttypes.JobStatus:
        """Gets the status of a batch from its ID."""
        return self._connection.get_job(batch_id).get_status()

    def _get_job_ids(self, batch_id: str) -> list[str]:
        """Gets all the job IDs within a batch."""
        raise NotImplementedError("There is no job IDs through this remote connection.")

    def _close_batch(self, batch_id: str) -> None:
        """Closes a batch using its ID."""
        raise NotImplementedError(  # pragma: no cover
            "Unable to close batch through this remote connection"
            "It can be cancelled using `cancel_batch`."
        )

    def cancel_batch(self, batch_id: str) -> None:
        """Cancels a batch using its ID."""
        return self._connection.get_job(batch_id).cancel()

    def delete_batch(self, batch_id: str) -> None:
        """Deletes the files of a batch using its ID."""
        return self._connection.get_job(batch_id).delete_files()

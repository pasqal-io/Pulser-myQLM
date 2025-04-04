"""Defines the PulserMyQLMBackend."""
from __future__ import annotations

import time
import typing

import pulser
import qat
from qat.comm.qlmaas.ttypes import QLMServiceException
from pulser.json.exceptions import DeserializeDeviceError

from pulser_myqlm.ising_aqpu import IsingAQPU
from pulser_myqlm.fresnel_qpu import FresnelQPU

class PulserQLMConnection(pulser.backend.remote.RemoteConnection):
    """The connection to a MyQLM QPU."""

    def __init__(self, connection: qat.qlmaas.QLMaaSConnection):
        self._connection = connection

    def supports_open_batch(self) -> bool:
        """Flag to confirm this class doesn't support open batch creation."""
        return False

    @staticmethod
    def _get_job_ids_from_batch(batch: qat.core.Batch) -> list[str]:
        """Generate the job IDs for a qat.core.Batch."""
        return [f"{i}" for (i, _) in enumerate(batch.jobs)]

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
            res_info.id,
            self,
            PulserQLMConnection._get_job_ids_from_batch(batch)
        )

    def fetch_available_devices(self) -> dict[str, pulser.devices.Device]:
        """Fetches the devices available through this connection."""
        qpus = self._connection.get_qpus()
        devices = {}
        for qpu in qpus:
            if "PasqalQPU" in qpu.name: 
                devices[qpu.name] = FresnelQPU(None).device
                continue
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
    ) -> typing.Sequence[qat.core.BatchResult | qat.core.Result]:
        """Fetches the results of a completed batch."""
        # Returns an error if job is not available
        results = self._connection.get_job(batch_id).get_result()
        if job_ids is None:
            return results
        batch_jobs_ids = self._get_job_ids(batch_id)
        sub_results = []
        for job_id in job_ids:
            try:
                job_idx_in_batch = batch_job_ids.index(job_id)
            except ValueError as e:
                raise ValueError(
                    f"Job id {job_id} is not among Job ids of the batch {batch_job_ids}"
                ) from e
            sub_results.append(results[job_idx_in_batch])
        return qat.core.BatchResult(results=sub_results, meta_data=results.meta_data)

    def _query_job_progress(
        self, batch_id: str
    ) -> typing.Mapping[str, tuple[qat.comm.qlmaas.ttypes.JobStatus, qat.core.Result | None]]:
        """Fetches the status and results of all the jobs in a batch.

        Unlike `_fetch_result`, this method does not raise an error if some
        jobs in the batch do not have results.

        It returns a dictionary mapping the job ID to its status and results.
        """
        status = self._get_batch_status(batch_id)
        job_ids = self._get_job_ids(batch_id)
        if status != qat.comm.qlmaas.ttypes.JobStatus.DONE:
            return {job_id:(status, None) for job_id in job_ids}
        results = self._connection.get_job(batch_id).get_result()
        return {job_ids[i]:(status, result) for (i, result) in enumerate(results.results)}

    def _get_batch_status(self, batch_id: str) -> qat.comm.qlmaas.ttypes.JobStatus:
        """Gets the status of a batch from its ID."""
        return self._connection.get_job(batch_id).get_status(human_readable=False)

    def _get_job_ids(self, batch_id: str) -> list[str]:
        """Gets all the job IDs within a batch."""
        qlm_batch = None
        for i in range(10):
            try: 
                qlm_batch = self._connection.get_job(batch_id).get_batch()
                break
            except QLMServiceException as e:
                time.sleep(1)
        if qlm_batch is None:
            raise QLMServiceException("Max retries performed")
        return PulserQLMConnection._get_job_ids_from_batch(qlm_batch)

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

"""Classes to wrap QLM connection with backoff."""

import logging
from typing import cast

import backoff
from qat.comm.qlmaas.ttypes import JobStatus
from qat.core import Job, Result
from qat.qlmaas import QLMaaSConnection
from qat.qlmaas.qpus import QLMaaSQPU
from qat.qlmaas.result import AsyncResult

from pulser_myqlm.constants import MAX_CONNECTION_ATTEMPTS_QLM

LOGGER = logging.getLogger(__name__)

backoff_decorator_qlm = backoff.on_exception(
    backoff.fibo,
    Exception,
    max_tries=MAX_CONNECTION_ATTEMPTS_QLM,
    max_value=60,
    logger=LOGGER,
)

# Wrapping is made for the purpose of retrying requests


class AsyncResultWrapper:
    """Wraps an AsyncResult class."""

    def __init__(self, result: AsyncResult) -> None:
        self.result: AsyncResult = result

    @backoff_decorator_qlm
    def get_id(self) -> str:
        """Get the job's ID."""
        return cast(str, self.result.get_info().id)

    @backoff_decorator_qlm
    def join(self) -> Result:
        """Wait until the job is done and returns the result of the job."""
        return self.result.join()

    @backoff_decorator_qlm
    def get_status(self) -> JobStatus:
        """Get the job's status."""
        return self.result.get_status(human_readable=False)

    @backoff_decorator_qlm
    def get_batch(self) -> Job:
        """Get the job's batch."""
        return self.result.get_batch()

    def get_result(self) -> Result:
        """Get the job's result. If the job is not available, an exception is raised."""
        return self.result.get_result()

    @backoff_decorator_qlm
    def delete_files(self) -> None:
        """Tries to delete job files."""
        self.result.delete_files()

    @backoff_decorator_qlm
    def cancel(self) -> None:
        """Tries to cancel the job."""
        self.result.cancel()


class QLMaaSQPUWrapper:
    """Wraps a QLMaaSQPU (a manager of a RemoteQPU)."""

    def __init__(self, qpu: QLMaaSQPU) -> None:
        self.qpu: QLMaaSQPU = qpu

    def submit(self, batch: Job) -> AsyncResultWrapper:
        """Submits a batch to the QPU. Creates an AsyncResultWrapper."""
        # no backoff to have Jobs submitted once
        return AsyncResultWrapper(self.qpu.submit(batch))

    @backoff_decorator_qlm
    def get_description(self) -> str:
        """Returns the description of the specs of the QPU."""
        return cast(str, self.qpu.get_specs().description)


class QLMClient:
    """Client for QLM."""

    def __init__(self, connection: QLMaaSConnection):
        self._connection = connection

    @backoff_decorator_qlm
    def _get_qpu_class(self, qpu_id: str) -> QLMaaSQPU:
        """Returns a QLMaaSQPU associated with the qpu_id in a QLMaaSConnection."""
        return self._connection.get_qpu(qpu_id)

    def get_qpu(self, qpu_id: str) -> QLMaaSQPUWrapper:
        """Returns a QLMQPUServer associated with the qpu_id in a QLMaaSConnection."""
        # No backoff because only subject to init errors (get_qpu_class has backoff)
        return QLMaaSQPUWrapper(self._get_qpu_class(qpu_id)())

    @backoff_decorator_qlm
    def list_qpu_names(self) -> list[str]:
        """Returns the names of the qpus available in the QLMaaSconnection."""
        return [qpu.name for qpu in self._connection.get_qpus()]

    def get_job(self, job_id: str) -> AsyncResultWrapper:
        """Gets the AsyncResultWrapper associated with a job_id in the connection."""
        return AsyncResultWrapper(self._connection.get_job(job_id))

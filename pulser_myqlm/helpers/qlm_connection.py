"""Classes to wrap QLM connection with backoff."""

import logging
from typing import cast

import backoff
from qat.comm.qlmaas.ttypes import JobStatus
from qat.core import Batch, BatchResult
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


class AsyncResultServer:
    """Wraps an AsyncResult class."""

    def __init__(self, result: AsyncResult) -> None:
        self.result: AsyncResult = result

    @backoff_decorator_qlm
    def get_id(self) -> str:
        """Get the job's ID."""
        return cast(str, self.result.get_info().id)

    @backoff_decorator_qlm
    def join(self) -> BatchResult:
        """Wait until the job is done and returns the result of the job."""
        return self.result.join()

    @backoff_decorator_qlm
    def get_status(self) -> JobStatus:
        """Get the job's status."""
        return self.result.get_status(human_readable=False)

    @backoff_decorator_qlm
    def get_batch(self) -> Batch:
        """Get the job's batch."""
        return self.result.get_batch()

    def get_result(self) -> BatchResult:
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


class QLMQPUServer:
    """Wraps a QLMaaSQPU (a manager of a RemoteQPU)."""

    def __init__(self, qpu: QLMaaSQPU):
        self.qpu: QLMaaSQPU = qpu

    @backoff_decorator_qlm
    def submit(self, batch: Batch) -> AsyncResultServer:
        """Submits a batch to the QPU. Creates an AsyncResultServer."""
        return AsyncResultServer(self.qpu.submit(batch))

    @backoff_decorator_qlm
    def get_description(self) -> str:
        """Returns the description of the specs of the QPU."""
        return cast(str, self.qpu.get_specs().description)


class QLMServer:
    """Static methods to call the QPU API."""

    @staticmethod
    @backoff_decorator_qlm
    def get_qpu(connection: QLMaaSConnection, qpu_id: str) -> QLMQPUServer:
        """Returns a QLMaaSQPU associated with the qpu_id in a QLMaaSConnection."""
        return QLMQPUServer(connection.get_qpu(qpu_id)())

    @staticmethod
    @backoff_decorator_qlm
    def get_qpus(connection: QLMaaSConnection) -> list[str]:
        """Returns the names of the qpus available in the QLMaaSconnection."""
        qpus_service_descr = connection.get_qpus()
        return [qpu.name for qpu in qpus_service_descr]

    @staticmethod
    def get_job(connection: QLMaaSConnection, job_id: str) -> AsyncResultServer:
        """Gets the AsyncResultServer associated with a job_id in the connection."""
        return AsyncResultServer(connection.get_job(job_id))

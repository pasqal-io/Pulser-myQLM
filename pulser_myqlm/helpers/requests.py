"""Functions to send requests with backoff mechanism."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, cast

import backoff
import requests

from pulser_myqlm.constants import MAX_CONNECTION_ATTEMPTS_QPU

logger = logging.getLogger(__name__)

backoff_decorator_qpu = backoff.on_exception(
    backoff.fibo,
    requests.exceptions.RequestException,
    max_tries=MAX_CONNECTION_ATTEMPTS_QPU,
    max_value=60,
    logger=logger,
)


class JobInfo:
    """Class to store the infos of a job running on the QPU."""

    def __init__(self, job_info: dict[str, Any]) -> None:
        self.job_info: dict[str, Any] = job_info

    def get_id(self) -> int:
        """Get the job's ID."""
        return cast(int, self.job_info["uid"])

    def get_status(self) -> str:
        """Gets the job's status."""
        return cast(str, self.job_info["status"])

    def get_counter_result(self) -> dict[str, int]:
        """When job is done, get the result as a counter."""
        return cast(dict[str, int], json.loads(self.job_info["result"])["counter"])

    def get_program_id(self) -> int:
        """Returns the program ID associated with the Job."""
        return cast(int, self.job_info["program_id"])


class PasqalQPUClient:
    """A client to communicate with the QPU's API.

    Args:
        base_uri: the IP address of the QPU.
        version: the version of its API.
    """

    def __init__(self, base_uri: str, version: str) -> None:
        self._base_uri = base_uri + "/" + version

    @property
    def base_uri(self) -> str:
        """Base URI of the QPU (IP/version)."""
        return self._base_uri

    def get_operational_status(self) -> str:
        """Gets QPU's operational status."""
        response = self._get_backoff("/system/operational")
        return cast(str, response.json()["data"]["operational_status"])

    def get_specs(self) -> str:
        """Gets the Device implemented by the QPU."""
        response = self._get_backoff("/system")
        return json.dumps(response.json()["data"]["specs"])

    def get_job_info(self, job_id: int, no_backoff: bool = False) -> JobInfo:
        """Gets information on a submitted job."""
        response = (self._get if no_backoff else self._get_backoff)(f"/jobs/{job_id}")
        return JobInfo(response.json()["data"])

    def get_program_status(self, program_id: int) -> str:
        """Gets the status of a program."""
        response = self._get_backoff(f"/programs/{program_id}")
        return json.dumps(response.json()["data"]["status"])

    def create_job(
        self, nb_run: int, abstract_sequence: str, batch_id: str | None = None
    ) -> JobInfo:
        """Create job on the QPU to run an abstract Sequence nb_run times."""
        # By default, submitting a job to the QPU cancels the previous job submitted
        pasqman_job_id = f"{uuid.uuid4()}"
        if batch_id is None:
            batch_id = pasqman_job_id
        logger.info(f"Creating pasqman_job_id {pasqman_job_id} for batch id {batch_id}")
        payload = {
            "nb_run": nb_run,
            "pulser_sequence": abstract_sequence,
            "context": {"batch_id": batch_id, "pasqman_job_id": pasqman_job_id},
        }
        response = self._post_backoff("/jobs", payload)
        return JobInfo(response.json()["data"])

    def cancel_job(self, job_info: JobInfo) -> None:
        """Terminates the execution of a given job ID."""
        program_id = job_info.get_program_id()
        try:
            program_status = self.get_program_status(program_id)
            logger.info(f"Program {program_id} has status {program_status}.")
            if program_status not in [
                '"ABORTED"',
                '"ABORTING"',
                '"ERROR"',
                '"MISSING_CALIBRATION"',
                '"DONE"',
                '"INVALID"',
            ]:
                self._delete_backoff(f"/programs/{program_id}")
                logger.info(
                    f"Program {program_id} is now "
                    f"{self.get_program_status(program_id)}."
                )
        except Exception:
            logger.exception(
                f"An error occured while trying to cancel program {program_id}."
            )

    @backoff_decorator_qpu
    def _get_backoff(self, suffix: str) -> requests.Response:
        """Sends a GET request to base_uri + suffix with backoff.

        Arg:
            suffix: The suffix to add after base_uri for the request.

        Returns:
            The requests.Response returned by the GET request.
        """
        return self._get(suffix)

    def _get(self, suffix: str) -> requests.Response:
        """Sends a GET request to base_uri + suffix.

        Arg:
            suffix: The suffix to add after base_uri for the request.

        Returns:
            The requests.Response returned by the GET request.
        """
        response = requests.get(self.base_uri + suffix)
        response.raise_for_status()
        return response

    @backoff_decorator_qpu
    def _post_backoff(self, suffix: str, data: dict | None = None) -> requests.Response:
        """Sends a POST request to base_uri + suffix with backoff.

        Arg:
            suffix: The suffix to add after base_uri for the request.
            data: The data to POST, as a JSON dictionnary.

        Returns:
            The requests.Response returned by the POST request.
        """
        response = requests.post(self.base_uri + suffix, json=data)
        response.raise_for_status()
        return response

    @backoff_decorator_qpu
    def _delete_backoff(self, suffix: str) -> requests.Response:
        """Sends a DELETE request to base_uri + suffix with backoff.

        Arg:
            suffix: The suffix to add after base_uri for the request.

        Returns:
            The requests.Response returned by the DELETE request.
        """
        response = requests.delete(self.base_uri + suffix)
        response.raise_for_status()
        return response

    @backoff_decorator_qpu
    def _put_backoff(self, suffix: str, data: dict | None = None) -> requests.Response:
        response = requests.put(self.base_uri + suffix, json=data)
        response.raise_for_status()
        return response

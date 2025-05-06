"""Functions to send requests with backoff mechanism."""

import json
import logging
from typing import Any, cast

import backoff
import requests

from pulser_myqlm.constants import MAX_CONNECTION_ATTEMPTS_QPU

LOGGER = logging.getLogger(__name__)

backoff_decorator_qpu = backoff.on_exception(
    backoff.fibo,
    requests.exceptions.RequestException,
    max_tries=MAX_CONNECTION_ATTEMPTS_QPU,
    max_value=60,
    logger=LOGGER,
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


class PasqalQPUClient:
    """Static methods to call the QPU API.

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
        """Gets QPU's operational status by calling the API using its base URI."""
        response = self._get_backoff("/system/operational")
        return cast(str, response.json()["data"]["operational_status"])

    def get_specs(self) -> str:
        """Gets the Device implemented by the QPU from its API."""
        response = self._get_backoff("/system")
        return json.dumps(response.json()["data"]["specs"])

    def get_job_info(self, job_id: int) -> JobInfo:
        """Gets information on a submitted job."""
        response = self._get(f"/jobs/{job_id}")
        return JobInfo(response.json()["data"])

    def post_job(self, nb_run: int, abstract_sequence: str) -> JobInfo:
        """Posts a Job to the QPU to run an abstract Sequence nb_run times."""
        payload = {"nb_run": nb_run, "pulser_sequence": abstract_sequence}
        response = self._post_backoff("/jobs", payload)
        return JobInfo(response.json()["data"])

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
    def _post_backoff(self, suffix: str, data: dict) -> requests.Response:
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

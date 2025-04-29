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

    def __init__(self, job_infos: dict[str, Any]) -> None:
        self.job_infos: dict[str, Any] = job_infos

    def get_id(self) -> int:
        """Get the job's ID."""
        return cast(int, self.job_infos["uid"])

    def get_status(self) -> str:
        """Gets the job's status."""
        return cast(str, self.job_infos["status"])

    def get_counter_result(self) -> dict[str, int]:
        """When job is done, get the result as a counter."""
        return cast(dict[str, int], json.loads(self.job_infos["result"])["counter"])


class QPUServer:
    """Static methods to call the QPU API."""

    @staticmethod
    def get_base_uri(base_uri: str, version: str) -> str:
        """Base URI of API from QPU's IP and API's version."""
        return base_uri + "/" + version

    @staticmethod
    def get_operational_status(base_uri: str) -> str:
        """Gets QPU's operational status by calling the API using its base URI."""
        response = get_url(url=base_uri + "/system/operational")
        return cast(str, response.json()["data"]["operational_status"])

    @staticmethod
    def get_specs(base_uri: str) -> str:
        """Gets the Device implemented by the QPU from its API."""
        response = get_url(base_uri + "/system")
        return json.dumps(response.json()["data"]["specs"])

    @staticmethod
    def request_job_info(base_uri: str, job_id: int) -> requests.Response:
        """Returns the response of a request for infos on a submitted job."""
        return requests.get(base_uri + f"/jobs/{job_id}")

    @staticmethod
    def get_job_info_from_response(response: requests.Response) -> JobInfo:
        """Converts a requests.Response asking about Job's infos into a JobInfo."""
        response.raise_for_status()
        return JobInfo(response.json()["data"])

    @staticmethod
    def post_job(base_uri: str, nb_run: int, abstract_sequence: str) -> JobInfo:
        """Posts a Job to the QPU to run an abstract Sequence nb_run times."""
        payload = {"nb_run": nb_run, "pulser_sequence": abstract_sequence}
        response = post_data_at_url(base_uri + "/jobs", payload)
        return JobInfo(response.json()["data"])


@backoff_decorator_qpu
def get_url(url: str) -> requests.Response:
    """Sends a GET request to the specified url.

    Arg:
        url: The url to request.

    Returns:
        The requests.Response returned by the GET request.
    """
    response = requests.get(url=url)
    response.raise_for_status()
    return response


@backoff_decorator_qpu
def post_data_at_url(url: str, data: dict) -> requests.Response:
    """Sends a POST request to the specified url.

    Arg:
        url: The url to request.
        data: The data to POST, as a JSON dictionnary.

    Returns:
        The requests.Response returned by the POST request.
    """
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response

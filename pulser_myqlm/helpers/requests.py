"""Functions to send requests with backoff mechanism."""

import logging

import backoff
import requests

from pulser_myqlm.constants import MAX_CONNECTION_ATTEMPTS_QPU

LOGGER = logging.getLogger("pulser-myqlm")

backoff_decorator_qpu = backoff.on_exception(
    backoff.fibo,
    requests.exceptions.RequestException,
    max_tries=MAX_CONNECTION_ATTEMPTS_QPU,
    max_value=60,
    logger=LOGGER,
)


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

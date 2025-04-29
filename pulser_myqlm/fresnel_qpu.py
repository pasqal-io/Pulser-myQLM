"""Defines Pulser AQPUs."""

from __future__ import annotations

import logging
import time
import warnings
from typing import cast

import requests
from pulser.devices._device_datacls import Device
from pulser.json.abstract_repr.deserializer import deserialize_device
from qat.comm.exceptions.ttypes import ErrorType, QPUException
from qat.core import HardwareSpecs, Job, Result
from qat.core.qpu import QPUHandler

from pulser_myqlm.constants import (
    DEFAULT_NUMBER_OF_SHOTS,
    JOB_POLLING_INTERVAL_SECONDS,
    QPU_POLLING_INTERVAL_SECONDS,
    TEMP_DEVICE,
)
from pulser_myqlm.helpers.deserialize_other import deserialize_other
from pulser_myqlm.helpers.requests import JobInfo, QPUServer
from pulser_myqlm.helpers.simulate_seq import simulate_seq
from pulser_myqlm.ising_aqpu import IsingAQPU

LOGGER = logging.getLogger(__name__)


class FresnelQPU(QPUHandler):
    r"""Fresnel Quantum Processing Unit.

    Connects to the API of the Fresnel QPU via a base_uri to send jobs to it.
    To deploy this QPU on a server, use its `serve` method. Any client can then
    access this QPU remotely using a `RemoteQPU` with correct port and IP.

    Args:
        base_uri: A string of shape 'https://myserver.com/api'. If None,
            pulser_simulation is used to simulate the Sequence.
        version: The version of the API to use, added at the end of the base URI.
        max_nbshots: The maximum number of shots per job. Default to
            `pulser_myqlm.devices.DEFAULT_NUMBER_OF_SHOTS`.
    """

    def __init__(
        self,
        base_uri: str | None,
        version: str = "latest",
        max_nbshots: int = DEFAULT_NUMBER_OF_SHOTS,
    ):
        super().__init__()
        self.max_nbshots = max_nbshots
        self.base_uri = (
            None if base_uri is None else QPUServer.get_base_uri(base_uri, version)
        )
        self.is_operational

    @property
    def is_operational(self) -> bool:
        """Returns whether or not the system is operational."""
        if self.base_uri is None:
            return True
        try:
            operational_status = QPUServer.get_operational_status(self.base_uri)
        except requests.exceptions.RequestException as e:
            raise QPUException(
                ErrorType.ABORT,
                message="Connection with API failed, make sure the address "
                f"{self.base_uri} is correct. Got error: {repr(e)}",
            )
        return operational_status == "UP"

    @property
    def deserialized_device(self) -> str:
        """The Device implemented by the hardware as a JSON string."""
        if self.base_uri is None:
            return TEMP_DEVICE.to_abstract_repr()
        try:
            device = QPUServer.get_specs(self.base_uri)
        except requests.exceptions.RequestException as e:
            raise QPUException(
                ErrorType.NONERESULT,
                "An error occured fetching the Device implemented by the QPU. "
                f"Got {repr(e)}",
            )
        return device

    @property
    def device(self) -> Device:
        """The current state of the Device implemented by the hardware."""
        if self.base_uri is None:
            return TEMP_DEVICE
        return cast(Device, deserialize_device(self.deserialized_device))

    def get_specs(self) -> HardwareSpecs:
        """Returns the HardwareSpecs of the QPU.

        An abstract representation of the Pulser Device is given in the 'description'
        of the HardwareSpecs. The Pulser Device can be obtained by using
        'pulser.json.abstract_repr.deserializer.deserialize_device'.
        """
        return HardwareSpecs(description=self.deserialized_device)

    def check_system(self, raise_error: bool = False) -> None:
        """Raises a warning or an error if the system is not operational.

        Deprecated. Not used in this class anymore.
        Maintained for backwards compatibility.
        """
        warnings.warn(
            "This method is deprecated, use poll_system instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.is_operational:
            msg = (
                "QPU not operational, please run calibration and validation of the "
                "devices prior to creating the FresnelQPU"
            )
            if raise_error:
                raise QPUException(ErrorType.ABORT, message=msg)
            warnings.warn(msg, UserWarning)

    def poll_system(self) -> None:
        """Polls QPU until it is operational."""
        msg = f"QPU not operational, will try again in {QPU_POLLING_INTERVAL_SECONDS}s"
        while not self.is_operational:
            warnings.warn(msg, UserWarning)
            time.sleep(QPU_POLLING_INTERVAL_SECONDS)

    def serve(
        self,
        port: int,
        host_ip: str = "localhost",
        server_type: str | None = None,
        **kwargs: str,
    ) -> None:
        """Runs the QPU inside a server.

        The QPU can only be run if it is operational.

        Args:
            port: The port on which to listen
            host_ip: The url on which to publish the API. Optional. Defaults to
                "localhost".
            server_type: Type of server. The different types of server are:
                "simple": single-thread server, accepts one connection at a time
                    (default server type)
                "threaded": multi-thread server, each connection starts a new thread
                "pool": multi-thread server, each connection runs in a thread, with
                    a maximum of 10 running threads
                "fork": multi-process server, each connection runs in a new process
        """
        self.poll_system()
        super().serve(port, host_ip, server_type, **kwargs)

    def poll_job_results(self, job_response: JobInfo) -> JobInfo:
        """Wait for the job to finish before querying results."""
        if self.base_uri is None:
            raise QPUException(
                ErrorType.ABORT,
                message="Results are only available if base_uri is defined",
            )
        job_id = job_response.get_id()
        while job_response.get_status() not in ["ERROR", "DONE"]:
            try:
                response = QPUServer.request_job_info(self.base_uri, job_id)
                job_response = QPUServer.get_job_info_from_response(response)
            except requests.ConnectionError as e:
                LOGGER.error(
                    f"Job request raised Connection Error: {repr(e)},"
                    + f" trying again in {JOB_POLLING_INTERVAL_SECONDS}s",
                )
            except requests.HTTPError as e:
                if 400 <= response.status_code < 500:
                    raise QPUException(
                        ErrorType.NONERESULT,
                        f"An error occured fetching your results: {repr(e)}",
                    )
                LOGGER.error(
                    f"Job request returned a {response.status_code} code: {repr(e)},"
                    + f" trying again in {JOB_POLLING_INTERVAL_SECONDS}s"
                )
            except requests.exceptions.RequestException as e:  # other requests error
                raise QPUException(
                    ErrorType.NONERESULT,
                    f"An error occured fetching your results: {repr(e)}",
                )
            time.sleep(JOB_POLLING_INTERVAL_SECONDS)

        # Check that the job submission went well
        if job_response.get_status() == "ERROR":
            raise QPUException(
                ErrorType.NONERESULT,
                "An error occured, check locally the Sequence before submitting or "
                "contact the support.",
            )

        return job_response

    def submit_job(self, job: Job) -> Result:
        """Submit a MyQLM job encapsulating a Pulser Sequence to the QPU.

        Args:
            job: The MyQLM Job to run. Must have an abstract Pulser Sequence under the
                key 'abstr_seq' of the dictionary serialized in Job.schedule._other. The
                Sequence must be compatible with FresnelDevice.
        """
        if job.schedule is None:
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message="FresnelQPU can only execute a schedule job.",
            )
        try:
            other_dict = deserialize_other(job.schedule._other)
        except ValueError as e:
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message=f"Failed at deserializing Job.Schedule._other. Got {repr(e)}",
            )
        seq = other_dict["seq"]
        # Validate that Sequence is compatible with FresnelDevice
        current_device = self.device
        try:
            if seq.device != current_device:
                seq = seq.switch_device(current_device, strict=True)
        except Exception as e:
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message="The Sequence in job.schedule._other['abstr_seq'] is not "
                f"compatible with the specs of the QPU: {self.device.specs}. Got "
                f"error {repr(e)}",
            )
        if current_device.requires_layout and seq.register.layout is None:
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message=(
                    "The Register of the Sequence in job.schedule._other['abstr_seq']"
                    " must be defined from a layout."
                ),
            )
        if (
            not current_device.accepts_new_layouts
            and not current_device.register_is_from_calibrated_layout(seq.register)
        ):
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message=(
                    "The Register of the Sequence in job.schedule._other['abstr_seq']"
                    " must be defined from one of the calibrated layouts."
                ),
            )
        if seq._empty_sequence:
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message=(
                    "The Sequence should contain at least one Pulse, added to a "
                    "declared channel."
                ),
            )
        modulation = other_dict.get("modulation", False)
        # Check that the system is operational
        self.poll_system()
        # Submit a job to the API
        nb_run = self.max_nbshots if not job.nbshots else job.nbshots
        if self.base_uri is None:
            pulser_results = simulate_seq(seq, modulation, nb_run)
            myqlm_result = IsingAQPU.convert_samples_to_result(pulser_results)
            return myqlm_result
        try:
            job_response = QPUServer.post_job(
                self.base_uri, nb_run, seq.to_abstract_repr()
            )
        except requests.exceptions.RequestException as e:
            raise QPUException(
                ErrorType.ABORT, message=f"Could not create job: Got {repr(e)}"
            )
        LOGGER.info(
            f"Job #{job_response.get_id()} created, status: {job_response.get_status()}"
        )

        job_response = self.poll_job_results(job_response)

        # Convert the output of the API into a MyQLM Result
        return IsingAQPU.convert_samples_to_result(job_response.get_counter_result())

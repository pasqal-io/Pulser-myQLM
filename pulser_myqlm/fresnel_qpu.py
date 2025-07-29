"""Defines Pulser AQPUs."""

from __future__ import annotations

import logging
import time
import warnings
from typing import cast
from datetime import datetime

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
    QPU_POLLING_TIMEOUT_SECONDS,
)
from pulser_myqlm.helpers.deserialize_other import deserialize_other
from pulser_myqlm.helpers.requests import JobInfo, PasqalQPUClient
from pulser_myqlm.helpers.simulate_seq import simulate_seq
from pulser_myqlm.ising_aqpu import IsingAQPU

logger = logging.getLogger(__name__)


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
            `pulser_myqlm.devices.DEFAULT_NUMBER_OF_SHOTS`. Number of shots per
            job will also be limited by the QPU itself.
    """

    def __init__(
        self,
        base_uri: str | None,
        version: str = "latest",
        max_nbshots: int = DEFAULT_NUMBER_OF_SHOTS,
    ):
        super().__init__()
        self._max_nbshots = max_nbshots
        self._qpu_client = (
            None if base_uri is None else PasqalQPUClient(base_uri, version)
        )
        # Check that base URI is correct and communication with QPU works
        self.is_operational

    @property
    def base_uri(self) -> None | str:
        """Base URI of the API of the QPU."""
        return None if self._qpu_client is None else self._qpu_client.base_uri

    @property
    def operational_status(self) -> str:
        """Returns the operational status of the system."""
        if self._qpu_client is None:
            return "UP"
        try:
            return self._qpu_client.get_operational_status()
        except requests.exceptions.RequestException as e:
            raise QPUException(
                ErrorType.ABORT,
                message="Connection with API failed, make sure the address "
                f"{self.base_uri} is correct. Got error: {repr(e)}",
            )

    @property
    def is_operational(self) -> bool:
        """Returns whether or not the system is operational."""
        return self.operational_status == "UP"

    @property
    def _deserialized_device(self) -> str:
        """The Device implemented by the hardware as a JSON string."""
        if self._qpu_client is None:
            return TEMP_DEVICE.to_abstract_repr()
        try:
            device = self._qpu_client.get_specs()
        except requests.exceptions.RequestException as e:
            raise QPUException(
                ErrorType.NONERESULT,
                "An error occured fetching the Device implemented by the QPU. "
                f"Got {repr(e)}",
            )
        return device

    @property
    def _device(self) -> Device:
        """The current state of the Device implemented by the hardware."""
        if self._qpu_client is None:
            return TEMP_DEVICE
        return cast(Device, deserialize_device(self._deserialized_device))

    def get_specs(self) -> HardwareSpecs:
        """Returns the HardwareSpecs of the QPU.

        An abstract representation of the Pulser Device is given in the 'description'
        of the HardwareSpecs. The Pulser Device can be obtained by using
        'pulser.json.abstract_repr.deserializer.deserialize_device'.
        """
        return HardwareSpecs(
            description=self._deserialized_device,
            meta_data={"operational_status": self.operational_status},
        )

    def _poll_system(self) -> None:
        """Polls QPU until it is operational."""
        msg = f"QPU not operational, will try again in {QPU_POLLING_INTERVAL_SECONDS}s"
        polling_start = datetime.now()
        while not self.is_operational:
            logger.warning(msg)
            warnings.warn(msg, UserWarning)
            time.sleep(QPU_POLLING_INTERVAL_SECONDS)
            if QPU_POLLING_TIMEOUT_SECONDS != -1 and (datetime.now() - polling_start).total_seconds() > QPU_POLLING_TIMEOUT_SECONDS:
                raise QPUException(
                    ErrorType.ABORT,
                    message=(
                        f"QPU not operational for more than {QPU_POLLING_TIMEOUT_SECONDS} seconds. Aborting. "
                        "Submit when the QPU's status is 'UP'. "
                        "Check `get_specs().meta_data['operational_status']`."
                    ),
                )
        logger.info("QPU is operational.")

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
        self._poll_system()
        super().serve(port, host_ip, server_type, **kwargs)

    def _wait_job_results(self, job_info: JobInfo) -> JobInfo:
        """Wait for the job to finish before querying results."""
        if self._qpu_client is None:
            raise QPUException(
                ErrorType.ABORT,
                message="Results are only available if base_uri is defined",
            )
        job_id = job_info.get_id()
        while (status := job_info.get_status()) not in ["ERROR", "DONE"]:
            logger.info(f"Current Job %s Status: %s", job_id, status)
            # We can't know how long processing the job will take on the QPU
            # We want to get the result, no matter QPU's availability
            # We poll the status of the job until termination ("ERROR" or "DONE")
            try:
                # No Backoff to handle errors separately
                job_info = self._qpu_client.get_job_info(job_id, no_backoff=True)
            except requests.ConnectionError:
                logger.exception(
                    "Job request raised Connection Error, trying again in "
                    f"{JOB_POLLING_INTERVAL_SECONDS}s",
                )
            except requests.HTTPError as e:
                if 400 <= e.response.status_code < 500:
                    raise QPUException(
                        ErrorType.NONERESULT,
                        f"An error occured fetching your results: {repr(e)}",
                    )
                logger.exception(
                    f"Job request returned a {e.response.status_code} code,"
                    + f" trying again in {JOB_POLLING_INTERVAL_SECONDS}s"
                )
            except requests.exceptions.RequestException as e:  # other requests error
                raise QPUException(
                    ErrorType.NONERESULT,
                    f"An error occured fetching your results: {repr(e)}",
                )
            time.sleep(JOB_POLLING_INTERVAL_SECONDS)

        # Check that the job submission went well
        if job_info.get_status() == "ERROR":
            raise QPUException(
                ErrorType.NONERESULT,
                "An error occured, check locally the Sequence before submitting or "
                "contact the support.",
            )

        return job_info

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
        if seq.is_parametrized():
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message=(
                    "Can't submit a parametrized sequence. Assign a value to each "
                    f"variable of the Sequence {list(seq.declared_variables.keys())}"
                    " prior to submission."
                ),
            )
        if seq.is_register_mappable():
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message=(
                    "Can't submit a sequence with a mappable register. Assign "
                    "coordinates to each qubit of the Sequence prior to submission."
                ),
            )
        # Validate that Sequence is compatible with FresnelDevice
        current_device = self._device
        try:
            if seq.device != current_device:
                seq = seq.switch_device(current_device, strict=True)
        except Exception as e:
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message="The Sequence in job.schedule._other['abstr_seq'] is not "
                f"compatible with the specs of the QPU: {self._device.specs}. Got "
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
        self._poll_system()
        # Submit a job to the API
        max_nb_run = (
            self._max_nbshots
            if current_device.max_runs is None
            else min(self._max_nbshots, current_device.max_runs)
        )
        nb_run = max_nb_run if not job.nbshots else job.nbshots
        if nb_run > max_nb_run:
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message=f"Too many runs asked. Max number of runs is {max_nb_run}.",
            )
        if self._qpu_client is None:
            pulser_results = simulate_seq(seq, modulation, nb_run)
            myqlm_result = IsingAQPU.convert_samples_to_result(pulser_results)
            return myqlm_result
        try:
            job_info = self._qpu_client.create_job(nb_run, seq.to_abstract_repr())
        except requests.exceptions.RequestException as e:
            raise QPUException(
                ErrorType.ABORT, message=f"Could not create job: Got {repr(e)}"
            )
        logger.info(
            f"Job #{job_info.get_id()} created, status: {job_info.get_status()}"
        )

        job_info = self._wait_job_results(job_info)
        counter = job_info.get_counter_result()
        logger.info(f"Job #{job_info.get_id()} done, got{counter}.")
        # Convert the output of the API into a MyQLM Result
        return IsingAQPU.convert_samples_to_result(counter)

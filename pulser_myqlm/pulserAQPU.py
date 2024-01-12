"""Defines Pulser AQPUs."""
from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from functools import cached_property
from typing import cast

import numpy as np
import requests
from pulser import Sequence, sampler
from pulser.channels import Rydberg
from pulser.devices._device_datacls import BaseDevice
from pulser.devices.interaction_coefficients import c6_dict
from pulser.register.base_register import BaseRegister
from pulser_simulation import QutipEmulator
from qat.core import Job, Observable, Result, Schedule, Term
from qat.core.qpu import CommonQPU, QPUHandler
from qat.core.variables import ArithExpression, Variable, cos, get_item, sin
from scipy.spatial.distance import cdist

from pulser_myqlm.devices import FresnelDevice


class IsingAQPU(QPUHandler):
    r"""Ising Analog Quantum Processing Unit.

    Device and register should respect a certain set of rules:
        - Device needs at least a Rydberg channel with global addressing.
        - Can only implement Ising Hamiltonians.
    The QPU used for the simulation must be able to run a time-dependent Job.

    Args:
        device: A device having a Rydberg.Global channel.
        register: A register defining the interactions between the atoms.
        qpu: The QPU to use to submit a job. Can be a QPU running locally or a
            RemoteQPU to run the job on a server.
    """

    device: BaseDevice
    register: BaseRegister
    qpu: CommonQPU | None

    def __init__(
        self, device: BaseDevice, register: BaseRegister, qpu: CommonQPU | None = None
    ) -> None:
        super().__init__()
        self.device = device
        self.check_channels_device(self.device)
        self.register = register
        self.set_qpu(qpu)

    @classmethod
    def from_sequence(cls, seq: Sequence, qpu: CommonQPU | None = None) -> IsingAQPU:
        """Creates an IsingAQPU with the device, register of a Sequence."""
        return cls(seq.device, seq.register, qpu)

    def check_channels_device(self, device: BaseDevice) -> None:
        """Check that the device has a Rydberg.Global channel."""
        for channel_name, channel_type in device.channels.items():
            if (
                isinstance(channel_type, Rydberg)
                and channel_type.addressing == "Global"
            ):
                self.channel = channel_name
                break
        else:
            raise ValueError(
                """
                Ising AQPU: the device should at least have
                a Rydberg channel with Global addressing.
                """
            )

    def set_qpu(self, qpu: CommonQPU | None = None) -> None:
        """Set the QPU to use to simulate jobs.

        Args:
            qpu: If None, pulser-simulation is used to simulate the Sequence
                associated to the Job. Otherwise, it can be a QPU running locally or a
                RemoteQPU to run the Job on a server.
        """
        self.qpu = qpu

    @property
    def nbqubits(self) -> int:
        """Number of qubits defined in the register."""
        return len(self.register.qubit_ids)

    @property
    def distances(self) -> np.ndarray:
        r"""Distances between each qubits (in :math:`\mu m`)."""
        positions = np.array(list(self.register.qubits.values()))
        return cast(np.ndarray, cdist(positions, positions, metric="euclidean"))

    @cached_property
    def c6_interactions(self) -> np.ndarray:
        r"""C6 Interactions between the qubits (in :math:`rad/\mu s`)."""
        interactions = np.zeros((self.nbqubits, self.nbqubits))
        same_atom = np.eye(self.nbqubits, dtype=bool)
        interactions[~same_atom] = (
            c6_dict[self.device.rydberg_level] / self.distances[~same_atom] ** 6
        )
        return interactions

    @cached_property
    def interaction_observables(self) -> Observable:
        """Computes the interaction terms of an Ising hamiltonian."""
        sum_c6_rows = np.sum(self.c6_interactions, axis=0)
        z_terms = [
            Term(self.c6_interactions[i, j] / 4, "ZZ", [i, j])
            for i in range(self.nbqubits)
            for j in range(i)
        ]
        z_terms.extend(
            [Term(sum_c6_rows[i] / 4, "Z", [i]) for i in range(self.nbqubits)]
        )
        return Observable(
            self.nbqubits,
            constant_coeff=np.sum(sum_c6_rows) / 8,
            pauli_terms=z_terms,
        )

    def pulse_observables(
        self,
        omega_t: ArithExpression | float,
        delta_t: ArithExpression | float,
        phi: ArithExpression | float,
    ) -> Observable:
        """Defines the terms associated to a pulse in Ising hamiltonian.

        Args:
            omega_t: Expression of the Rabi frequency (in rad/µs).
            delta_t: Expression of the detuning (in rad/µs).
            phi: Expression of the phase (in rad).

        Returns:
            The corresponding ising hamiltonian.
        """
        terms = {
            "X": omega_t / 2.0 * cos(phi),
            "Y": -omega_t / 2.0 * sin(phi),
            "Z": -delta_t / 2.0,
        }
        return Observable(
            self.nbqubits,
            pauli_terms=[
                Term(coeff, op, [i])
                for i in range(self.nbqubits)
                for op, coeff in terms.items()
            ],
        )

    def hamiltonian(
        self,
        omega_t: ArithExpression | float,
        delta_t: ArithExpression | float,
        phi: ArithExpression | float,
    ) -> Observable:
        """Defines an Ising hamiltonian from a pulse.

        Args:
            omega_t: Expression of the Rabi frequency (in rad/µs).
            delta_t: Expression of the detuning (in rad/µs).
            phase: Expression of the phase (in rad).

        Returns:
            The corresponding ising hamiltonian.
        """
        return (
            self.pulse_observables(omega_t, delta_t, phi) + self.interaction_observables
        )

    @classmethod
    def convert_sequence_to_schedule(
        cls,
        seq: Sequence,
        modulation: bool = False,
    ) -> Schedule:
        """Converts a Pulser Sequence to a Myqlm Schedule.

        For a Sequence with max one declared channel, that channel being Rydberg.Global.
        Samples the Sequence, eventually modulates it using its modulation bandwidth.
        Outputs a time-dependent Ising Hamiltonian in a Schedule.

        Args:
            seq: The Pulser Sequence to convert.
            modulation: Whether the Schedule should contain modulated samples or not.
                Modulation is performed using the modulation bandwidth of the channel,
                it is used in simulations to model more accurately the behaviour of the
                channel.

        Returns:
            schedule: A MyQLM Schedule representing a time-dependent Ising hamiltonian.
        """
        qpu = cls.from_sequence(seq)
        # Check that the sequence has only one global channel declared
        declared_channel = list(seq.declared_channels.values())
        if len(declared_channel) > 1:
            raise ValueError("More than one channel declared.")
        elif len(declared_channel) == 1:
            ch_obj = declared_channel[0]
            if not isinstance(ch_obj, Rydberg):
                raise TypeError("Declared channel is not Rydberg.")
            elif ch_obj.addressing != "Global":
                raise TypeError("Declared channel is not Rydberg.Global.")
        else:
            # empty schedule if empty sequence
            return Schedule()

        ch_name = list(seq.declared_channels.keys())[0]
        # Sample the sequence
        ch_sample = sampler.sample(seq, modulation).channel_samples[ch_name]
        tmax = ch_sample.duration
        t = Variable("t")
        # Convert the samples of amplitude, detuning and phase to ArithExpression.
        omega_t = get_item(list(ch_sample.amp), t)
        delta_t = get_item(list(ch_sample.det), t)
        phi_t = get_item(list(ch_sample.phase), t)
        # Drive values are Ising hamiltonian at each time-step
        sch = Schedule(
            [
                (1, qpu.pulse_observables(omega_t, delta_t, phi_t)),
                (1, qpu.interaction_observables),
            ],
            tmax=tmax,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Register serialization", UserWarning)
            sch._other = {"abstr_seq": seq.to_abstract_repr(), "modulation": modulation}
        return sch

    @classmethod
    def convert_sequence_to_job(
        cls,
        seq: Sequence,
        nbshots: int = 0,
        modulation: bool = False,
    ) -> Job:
        """Converts a Pulser Sequence to a Myqlm Job.

        For a Sequence with max one declared channel, that channel being Rydberg.Global.
        Samples the Sequence, eventually modulates it using its modulation bandwidth.
        Outputs a Job with a time-dependent Ising Hamiltonian in its Schedule.

        Args:
            seq: The Pulser Sequence to convert.
            nbshots: The number of shots to perform. Default to 0 (asks for the max
                number of shots that can be performed on the qpu).
            modulation: Whether the Schedule of the Job should contain modulated
                samples. Modulation is performed using the modulation bandwidth of the
                channel, it is used in simulations to model more accurately the
                behaviour of the channel.

        Returns:
            job: a Job with a time-dependent Ising hamiltonian in its schedule.
        """
        schedule = cls.convert_sequence_to_schedule(seq, modulation)
        return schedule.to_job(nbshots=nbshots)

    @staticmethod
    def convert_pulser_samples(pulser_samples: Counter | dict[str, int]) -> Result:
        """Converts the output of a sampling into a myqlm Result.

        Args:
            pulser_samples: A dictionary of strings describing the measured states
                and their respective counts.

        Returns:
            Result: A myqlm Result associating each state with
                its frequency of occurence in pulser_samples.
        """
        n_samples = sum(pulser_samples.values())
        # Associates to each measured state its frequency of occurence
        myqlm_result = Result()
        for state, counts in pulser_samples.items():
            myqlm_result.add_sample(int(state, 2), probability=counts / n_samples)
        return myqlm_result

    def submit_job(self, job: Job, **kwargs) -> Result:  # type: ignore
        """Submit a MyQLM job to the QPU.

        If no QPU has been provided, simulation of the Pulser Sequence associated with
        the MyQLM Job is performed using pulser_simulation. Default number of shots is
        2000. Pulser Sequence should be provided inside a dict in job.schedule._other,
        under the key "abstr_seq". Whether to simulate using modulated samples of the
        Sequence is defined by the value associated to "modulation" in this dictionary.

        Args:
            job: the MyQLM Job to run.

        Kwargs:
            Kwargs of the defined QPU.
        """
        if self.qpu is not None:
            return self.qpu.submit_job(job, **kwargs)
        if (
            job.schedule._other is None
            or not isinstance(job.schedule._other, dict)
            or "abstr_seq" not in job.schedule._other
        ):
            raise ValueError(
                "An abstract representation of the Sequence must be associated with the"
                " 'abstr_seq' key of the dictionary in Job.schedule._other."
            )
        seq = Sequence.from_abstract_repr(job.schedule._other["abstr_seq"])
        emulator = QutipEmulator.from_sequence(
            seq,
            with_modulation=False
            if "modulation" not in job.schedule._other
            else job.schedule._other["modulation"],
        )
        pulser_samples = emulator.run().sample_final_state(
            2000 if not job.nbshots else job.nbshots
        )
        return self.convert_pulser_samples(pulser_samples)


class FresnelQPU(QPUHandler):
    r"""Fresnel Quantum Processing Unit.

    Connects to the API of the Fresnel QPU via a base_uri to send jobs to it.
    To deploy this QPU on a server, use its `serve` method. Any client can then
    access this QPU remotly using a `RemoteQPU` with correct port and IP.

    Args:
        base_uri: A string of shape 'https://myserver.com/api'.
        version: The version of the API to use, added at the end of the base URI.
        max_nbshots: The maximum number of shots per job. Default to 2000.
    """

    def __init__(
        self,
        base_uri: str,
        version: str = "latest",
        max_nbshots: int = 2000,
    ):
        super().__init__()
        self.max_nbshots = max_nbshots
        self.base_uri = base_uri + "/" + version
        self.device: BaseDevice = FresnelDevice
        self.is_operational  # Check that base_uri is correct

    @property
    def is_operational(self) -> bool:
        """Returns whether or not the system is operational."""
        response = requests.get(url=self.base_uri + "/system/operational")
        if response.status_code != 200:
            raise ValueError(
                "Connection with API failed, make sure the address "
                f"{self.base_uri} is correct."
            )
        return cast(str, response.json()["data"]["operational_status"]) == "UP"

    def check_system(self) -> None:
        """Check that the system is operational."""
        if not self.is_operational:
            raise OSError(
                "QPU not operational, please run calibration and validation of the "
                "devices prior to creating the FresnelQPU"
            )

    def serve(
        self, port: int, host_ip: str = "localhost", server_type: str | None = None
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
        self.check_system()
        super().serve(port, host_ip, server_type)

    def submit_job(self, job: Job) -> Result:
        """Submit a MyQLM job encapsulating a Pulser Sequence to the QPU.

        Args:
            job: The MyQLM Job to run. Must have an abstract Pulser Sequence under the
                key 'abstr_seq' of the dictionary Job.schedule._other.
        """
        if (
            job.schedule._other is None
            or not isinstance(job.schedule._other, dict)
            or "abstr_seq" not in job.schedule._other
        ):
            raise ValueError(
                "An abstract representation of the Sequence must be associated with the"
                " 'abstr_seq' key of the dictionary in Job.schedule._other."
            )
        # Check that the system is operational
        self.check_system()

        # Submit a job to the API
        payload = {
            "nb_run": self.max_nbshots if not job.nbshots else job.nbshots,
            "pulser_sequence": job.schedule._other["abstr_seq"],
        }
        response = requests.post(self.base_uri + "/jobs", json=payload)
        if response.status_code != 200:
            raise Exception("Could not create job", response.text)
        job_response = response.json()["data"]
        print(f"Job #{job_response['uid']} created, status: {job_response['status']}")

        # Wait for the job to finish before querying results
        while response.status_code == 200 and job_response["status"] not in [
            "ERROR",
            "DONE",
        ]:
            assert job_response["status"] in ["PENDING", "RUNNING"]
            time.sleep(10)
            response = requests.get(self.base_uri + f"/jobs/{job_response['uid']}")
            if response.status_code != 200:
                print("Error while getting job status, exiting loop", response.text)
                continue
            job_response = response.json()["data"]

        # Check that the job submission went well
        if job_response["status"] == "ERROR":
            raise RuntimeError(
                "An error occured, check locally the Sequence before submitting or "
                "contact the support."
            )
        assert job_response["status"] == "DONE"
        # Convert the output of the API into a MyQLM Result
        pulser_json_result = job_response["result"]
        if pulser_json_result is None:
            return Result()
        pulser_result = json.loads(pulser_json_result)
        return IsingAQPU.convert_pulser_samples(pulser_result)

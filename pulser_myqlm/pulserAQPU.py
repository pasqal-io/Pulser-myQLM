"""Defines Pulser AQPUs."""
from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from functools import cached_property
from typing import Optional, cast

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


class PulserAQPU(QPUHandler):
    r"""Base class of a Pulser Analog Quantum Processing Unit.

    A PulserAQPU is defined by a Device and a Register.
    The Observable given in the job are in :math:`\hbar rad/\mu s`.

    Attributes:
        device: The device used for the computations.
        register: The register used.
        qpu: The QPU to use when submitting a Job. By default or if the QPU is None,
            pulser-simulation is used to simulate the Sequence associated to the Job
            locally. Otherwise, it can be a QPU running locally or a RemoteQPU to
            run the Job on a server.
    """

    device: BaseDevice
    register: BaseRegister
    qpu: CommonQPU | None

    def __init__(
        self, device: BaseDevice, register: BaseRegister, qpu: CommonQPU | None = None
    ) -> None:
        super().__init__()
        self.device = device
        self.register = register
        self.qpu = qpu

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
        positions = self.register._coords
        return cast(np.ndarray, cdist(positions, positions, metric="euclidean"))

    @staticmethod
    def convert_pulser_samples(
        pulser_samples: Counter | dict[str, int],
    ) -> Result:
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

    def submit_job(self, job: Job) -> None:
        """Should be implemented for each qpu."""
        raise NotImplementedError(
            "Submit job only implemented for hardware-specific qpus, not PulserAQPU."
        )


class IsingAQPU(PulserAQPU):
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

    def __init__(
        self, device: BaseDevice, register: BaseRegister, qpu: CommonQPU | None = None
    ) -> None:
        super().__init__(device=device, register=register, qpu=qpu)
        self.check_channels_device(self.device)

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
        extended_duration: Optional[int] = None,
    ) -> Schedule:
        """Converts a Pulser Sequence to a Myqlm Schedule.

        For a sequence with max one declared channel, that channel being Rydberg.global
        Samples the channel, eventually extends its duration or modulates it.
        Outputs a time dependent hamiltonian defined at each time-step using heaviside.

        Args:
            seq: the sequence to convert.
            modulation: Whether to modulate the samples.
            extended_duration: duration by which to extend the samples.

        Returns:
            schedule: a sum of time-dependent Ising hamiltonians.
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
        ch_sample = sampler.sample(seq, modulation, extended_duration).channel_samples[
            ch_name
        ]
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
        extended_duration: Optional[int] = None,
    ) -> Job:
        """Converts a Pulser Sequence to a Myqlm Job.

        Args:
            seq: The Pulser Sequence to convert.
            nbshots: The number of shots to perform. Default to 0 (max number of shots
                that can be performed on the qpu).
            modulation: Whether to modulate the samples of the Sequence.
            extended_duration: If defined, extends the samples duration to the
                desired value.
        """
        schedule = cls.convert_sequence_to_schedule(seq, modulation, extended_duration)
        return schedule.to_job(nbshots=nbshots)

    def submit_job(self, job: Job) -> Result:
        """Submit a MyQLM job to the QPU.

        If no QPU has been provided, simulation of the sequence associated with the job
        is performed via pulser_simulation. Default number of shots is 2000.
        """
        if self.qpu is None:
            if job.schedule._other is None:
                raise ValueError(
                    "If no QPU is defined, provide an abstract representation of the "
                    "Sequence associated to the Job in the Job.schedule._other. "
                    "Otherwise, define a QPU using `set_qpu`."
                )
            seq = Sequence.from_abstract_repr(job.schedule._other["abstr_seq"])
            emulator = QutipEmulator.from_sequence(
                seq, with_modulation=job.schedule._other["modulation"]
            )
            pulser_samples = emulator.run().sample_final_state(
                2000 if not job.nbshots else job.nbshots
            )
            return self.convert_pulser_samples(pulser_samples)
        return self.qpu.submit_job(job)


class FresnelQPU(QPUHandler):
    r"""Fresnel Quantum Processing Unit.

    Args:
        base_uri: A string of shape 'https://myserver.com/api'.
        version: The version of the API to use, added at the end of the base URI.
        max_nbshots: The maximum number of shots per job. Default to 2000.
    """

    def __init__(
        self,
        base_uri,
        version="latest",
        max_nbshots: int = 2000,
        cancel_previous_job: bool = False,
    ):
        super.__init__(self)
        # api-endpoint
        self.max_nbshots = max_nbshots
        self.cancel_previous_job = cancel_previous_job
        self.base_uri = base_uri + "/" + version
        self.update_system()
        self.print_system()
        self.check_system()

    def update_system(self):
        self.system = requests.get(url=self.base_uri + "/system").json()

    def print_system(self):
        print("Using QPU: ", self.system["model_name"])
        print("With ID: ", self.system["serial_number"])
        print("QPU mode: ", self.system["mode"])
        print("QPU status: ", self.system["status"])
        print("Time to idle: ", self.system["time_to_idle"])

    def check_system(self):
        if self.system["mode"] != "MODE_OPERATION":
            raise OSError(
                "QPU not operational, please run calibration and validation of the "
                "devices prior to creating the FresnelQPU"
            )
        if self.system["status"] == "FAILURE":
            raise OSError(
                "The QPU has failed its last job and can't recover. Operator "
                "intervention is required."
            )

    def submit_job(self, job: Job, cancel_previous_job: bool | None = None):
        """Submit a MyQLM job encapsulating a Pulser Sequence to the QPU."""
        if job.schedule._other is None:
            raise ValueError(
                "You must provide an abstract representation of the Sequence associated"
                " to the Job in the Job.schedule._other."
            )
        if "abstr_seq" not in job.schedule._other:
            raise ValueError(
                "An abstract representation of the Sequence to run must be provided under "
                "the key 'abstr_seq' in job.schedule._other."
            )
        self.update_system()
        self.print_system()
        self.check_system()
        api_job = {
            "nb_run": self.max_nbshots if not job.nbshots else job.nbshots,
            "pulser": job.schedule._other["abstr_seq"],
            "cancel_previous_job": self.cancel_previous_job
            if cancel_previous_job is not None
            else cancel_previous_job,
        }
        api_job_return = requests.post(self.base_uri + "/job", json=api_job).json()
        if api_job_return["status"] == "ERROR":
            raise RuntimeError(
                "An error occured, check locally the Sequence before submitting."
            )
        assert api_job_return["client_id"] == 201
        self.update_system()
        os.wait(self.system["time_to_idle"])
        api_job_out = requests.get(
            self.base_uri + f"/jobs/{api_job_return['uid']}"
        ).json()
        assert api_job_out["status"] == "DONE"
        pulser_json_result = api_job_out["result"]
        if pulser_json_result is None:
            return Result()
        pulser_result = json.loads(pulser_json_result)
        return PulserAQPU.convert_pulser_samples(pulser_result)

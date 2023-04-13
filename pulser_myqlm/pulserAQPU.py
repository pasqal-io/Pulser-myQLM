"""Defines Pulser AQPUs."""
from __future__ import annotations

from collections import Counter
from typing import Optional, cast

import numpy as np
from pulser import Sequence, sampler
from pulser.channels import Rydberg
from pulser.devices._device_datacls import BaseDevice
from pulser.devices.interaction_coefficients import c6_dict
from pulser.register.base_register import BaseRegister
from qat.core import Job, Observable, Result, Schedule, Term
from qat.core.qpu import QPUHandler
from qat.core.variables import ArithExpression, Variable, cos, sin
from scipy.spatial.distance import cdist

from pulser_myqlm.myqlmtools import Pheaviside, PSchedule


class PulserAQPU(QPUHandler):
    r"""Base class of a Pulser Analog Quantum Processing Unit.

    A PulserAQPU is defined by a Device and a Register.
    The Observable given in the job are in :math:`\hbar rad/\mu s`.

    Attributes:
        device: The device used for the computations.
        register: The register used.
    """

    device: BaseDevice
    register: BaseRegister

    def __init__(self, device: BaseDevice, register: BaseRegister) -> None:
        super().__init__()
        self.device = device
        self.register = register

    @property
    def nbqubits(self) -> int:
        """Number of qubits defined in the register."""
        return len(self.register.qubit_ids)

    @property
    def distances(self) -> np.ndarray:
        r"""Distances between each qubits (in :math:`\mu m`)."""
        positions = self.register._coords
        return cast(np.ndarray, cdist(positions, positions, metric="euclidean"))

    def convert_pulser_samples(
        self,
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

    Args:
        device: A device having a Rydberg.Global channel.
        register: A register defining the interactions between the atoms.
    """

    def __init__(self, device: BaseDevice, register: BaseRegister) -> None:
        super().__init__(device=device, register=register)
        self.check_channels_device(self.device)

    @classmethod
    def from_sequence(cls, seq: Sequence) -> IsingAQPU:
        """Creates an IsingAQPU with the device, register of a Sequence."""
        return cls(seq.device, seq.register)

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

    @property
    def c6_interactions(self) -> np.ndarray:
        r"""C6 Interactions between the qubits (in :math:`rad/\mu s`)."""
        interactions = np.zeros((self.nbqubits, self.nbqubits))
        same_atom = np.eye(self.nbqubits, dtype=bool)
        interactions[~same_atom] = (
            c6_dict[self.device.rydberg_level] / self.distances[~same_atom] ** 6
        )
        return interactions

    @property
    def interaction_observables(self) -> Observable:
        """Computes the interaction terms of an Ising hamiltonian."""
        c6_interactions = self.c6_interactions
        terms = [
            Observable(
                self.nbqubits,
                pauli_terms=[
                    Term(1.0, "II", [i, j]),
                    Term(1.0, "IZ", [i, j]),
                    Term(1.0, "ZI", [i, j]),
                    Term(1.0, "ZZ", [i, j]),
                ],
            )
            * c6_interactions[i, j]
            for i in range(self.nbqubits)
            for j in range(i)
        ]
        return sum(terms) / 4.0

    def pulse_observables(
        self,
        omega_t: ArithExpression | float | np.ndarray,
        delta_t: ArithExpression | float | np.ndarray,
        phi: float | np.ndarray,
    ) -> Observable | np.ndarray:
        """Defines the terms associated to a pulse in Ising hamiltonian.

        Args:
            omega_t: Expression of the Rabi frequency (in rad/µs).
            delta_t: Expression of the detuning (in rad/µs).
            phi: Value of the phase (in rad).

        Returns:
            The corresponding ising hamiltonian.
        """
        X_observables = np.array(
            [
                Observable(self.nbqubits, pauli_terms=[Term(1.0, "X", [i])])
                for i in range(self.nbqubits)
            ]
        )
        Y_observables = np.array(
            [
                Observable(self.nbqubits, pauli_terms=[Term(1.0, "Y", [i])])
                for i in range(self.nbqubits)
            ]
        )
        Z_observables = np.array(
            [
                Observable(self.nbqubits, pauli_terms=[Term(1.0, "Z", [i])])
                for i in range(self.nbqubits)
            ]
        )

        return omega_t / 2.0 * (
            cos(phi) * np.sum(X_observables) - sin(phi) * np.sum(Y_observables)
        ) - delta_t / 2.0 * np.sum(Z_observables)

    def hamiltonian(
        self,
        omega_t: ArithExpression | float | np.ndarray,
        delta_t: ArithExpression | float | np.ndarray,
        phi: float | np.ndarray,
    ) -> Observable | np.ndarray:
        """Defines an Ising hamiltonian from a pulse.

        Args:
            omega_t: Expression of the Rabi frequency (in rad/µs).
            delta_t: Expression of the detuning (in rad/µs).
            phase: Value of the phase (in rad).

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
        asPSchedule: bool = True,
    ) -> Schedule | PSchedule:
        """Converts a Pulser Sequence to a Myqlm Schedule.

        For a sequence with max one declared channel, that channel being Rydberg.global
        Samples the channel, eventually extends its duration or modulates it.
        Outputs a time dependent hamiltonian defined at each time-step using heaviside.

        Args:
            seq: the sequence to convert.
            modulation: Whether to modulate the samples.
            extended_duration: duration by which to extend the samples.
            asPSchedule: Outputs a PSchedule if True, a Schedule otherwise.

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
            return PSchedule() if asPSchedule else Schedule()
        ch_name = list(seq.declared_channels.keys())[0]
        # Sample the sequence
        ch_sample = sampler.sample(seq, modulation, extended_duration).channel_samples[
            ch_name
        ]
        tmax = ch_sample.duration
        t = Variable("t")
        # Drive coefficients represent each time-step
        drive_coeffs = np.array([Pheaviside(t, ti, ti + 1) for ti in range(tmax)])
        # Drive values are Ising hamiltonian at each time-step
        drive_values = (
            qpu.pulse_observables(
                np.array(ch_sample.amp),
                np.array(ch_sample.det),
                np.array(ch_sample.phase),
            )
            + qpu.interaction_observables
        )

        drive = np.column_stack((drive_coeffs, drive_values))

        return (
            PSchedule(drive, tmax=tmax) if asPSchedule else Schedule(drive, tmax=tmax)
        )

    @classmethod
    def convert_sequence_to_job(
        cls,
        seq: Sequence,
        modulation: bool = False,
        extended_duration: Optional[int] = None,
    ) -> Job:
        """Converts a Pulser Sequence to a Myqlm Job."""
        schedule = cls.convert_sequence_to_schedule(seq, modulation, extended_duration)
        return schedule.to_job()

    def submit_job(self, job: Job) -> None:
        """Not implemented yet."""
        pass

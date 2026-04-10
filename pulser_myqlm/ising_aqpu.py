"""Defines Pulser AQPUs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import cached_property
from typing import Any, Union, cast

import numpy as np
from pulser import Sequence, sampler
from pulser.channels import Rydberg
from pulser.devices._device_datacls import COORD_PRECISION, BaseDevice
from pulser.devices.interaction_coefficients import c6_dict
from pulser.register.base_register import BaseRegister
from qat.comm.exceptions.ttypes import ErrorType, QPUException
from qat.core import Batch, BatchResult, Job, Observable, Result, Schedule, Term
from qat.core.qpu import CommonQPU, QPUHandler
from qat.core.variables import ArithExpression, Variable, cos, get_item, sin
from qat.lang.AQASM.bits import QRegister
from qat.qlmaas.qpus import QLMaaSQPU
from scipy.spatial.distance import cdist

from pulser_myqlm.helpers.deserialize_other import deserialize_other
from pulser_myqlm.helpers.simulate_seq import simulate_seq

QPUType = Union[None, CommonQPU, QLMaaSQPU]


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
    qpu: QPUType

    def __init__(
        self, device: BaseDevice, register: BaseRegister, qpu: QPUType = None
    ) -> None:
        super().__init__()
        for test_value in [
            (device, "device", BaseDevice),
            (register, "register", BaseRegister),
        ]:
            if not isinstance(test_value[0], test_value[2]):
                raise TypeError(
                    f"The provided {test_value[1]} must be of type {test_value[2]},"
                    f" not {type(test_value[0])}"
                )
        self.device = device
        self.check_channels_device(self.device)
        self.register = register
        self.set_qpu(qpu)

    @classmethod
    def from_sequence(cls, seq: Sequence, qpu: QPUType = None) -> IsingAQPU:
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

    def set_qpu(self, qpu: QPUType = None) -> None:
        """Set the QPU to use to simulate jobs.

        Args:
            qpu: If None, pulser-simulation is used to simulate the Sequence
                associated to the Job. Otherwise, it can be a QPU running locally or a
                RemoteQPU to run the Job on a server.
        """
        if qpu is not None and not (
            isinstance(qpu, CommonQPU) or isinstance(qpu, QLMaaSQPU)
        ):
            raise TypeError(
                "The provided qpu must be None, a `CommonQPU` instance (QPUHandler,"
                " RemoteQPU, ...) or a QLMaaSQPU (from qlmaas.qpus)."
            )
        self.qpu = qpu

    @property
    def nbqubits(self) -> int:
        """Number of qubits defined in the register."""
        return len(self.register.qubit_ids)

    @property
    def distances(self) -> np.ndarray:
        r"""Distances between each qubits (in :math:`\mu m`)."""
        positions = np.array(list(self.register.qubits.values()))
        return np.round(
            cast(np.ndarray, cdist(positions, positions, metric="euclidean")),
            COORD_PRECISION,
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
            [Term(-sum_c6_rows[i] / 4, "Z", [i]) for i in range(self.nbqubits)]
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
            "Y": omega_t / 2.0 * sin(phi),
            "Z": delta_t / 2.0,
        }
        return Observable(
            self.nbqubits,
            constant_coeff=-delta_t / 2.0,
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
        Outputs a time-dependent Ising Hamiltonian in rad/µs in a Schedule. The time in
        the Hamiltonian is defined in µs. The Hamiltonian is defined every 0.001µs.

        Args:
            seq: The Pulser Sequence to convert (times are defined in ns).
            modulation: Whether the Schedule should contain modulated samples or not.
                Modulation is performed using the modulation bandwidth of the channel,
                it is used in simulations to model more accurately the behaviour of the
                channel.

        Returns:
            schedule: A MyQLM Schedule representing a time-dependent Ising hamiltonian.
        """
        qpu = cls.from_sequence(seq)
        _other = json.dumps(
            {"abstr_seq": seq.to_abstract_repr(), "modulation": modulation}
        ).encode("utf-8")
        # Check that the sequence has only one global channel declared
        declared_channel = list(seq.declared_channels.values())
        if len(declared_channel) > 1:
            raise ValueError("More than one channel declared.")
        elif len(declared_channel) == 0:
            # Schedule with interaction terms if empty sequence
            sch = Schedule([(1, qpu.interaction_observables)], tmax=0)
            sch._other = _other
            return sch
        # one single channel declared
        ch_obj = declared_channel[0]
        if not isinstance(ch_obj, Rydberg):
            raise TypeError("Declared channel is not Rydberg.")
        elif ch_obj.addressing != "Global":
            raise TypeError("Declared channel is not Rydberg.Global.")

        ch_name = list(seq.declared_channels.keys())[0]
        # Sample the sequence
        ch_sample = sampler.sample(seq, modulation).channel_samples[ch_name]
        t = Variable("t")  # in µs
        # Convert the samples of amplitude, detuning and phase to ArithExpression.
        omega_t = get_item(ch_sample.amp.tolist(), t * 1000)  # samples are every ns
        delta_t = get_item(ch_sample.det.tolist(), t * 1000)  # samples are every ns
        phi_t = get_item(ch_sample.phase.tolist(), t * 1000)  # samples are every ns
        # Drive values are Ising hamiltonian at each time-step
        sch = Schedule(
            [
                (1, qpu.pulse_observables(omega_t, delta_t, phi_t)),
                (1, qpu.interaction_observables),
            ],
            tmax=ch_sample.duration / 1000,  # in µs
        )
        sch._other = _other
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
        Outputs a time-dependent Ising Hamiltonian in rad/µs in a Schedule. The time in
        the Hamiltonian is defined in µs. The Hamiltonian is defined every 0.001µs.

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
    def convert_samples_to_result(
        result_samples: Counter | dict[str, int], aggregate_samples: bool = True
    ) -> Result:
        """Converts the output of a sampling into a MyQLM Result.

        Args:
            result_samples: A dictionary of strings describing the measured states
                and their respective counts.
            aggregate: If False, the samples are not aggregated ie the myQLM
                Result has n_samples Samples of frequency 1/n_samples.

        Returns:
            Result: A myqlm Result associating each state with
                its frequency of occurence in pulser_samples.
        """
        n_samples = int(sum(result_samples.values()))
        # Associates to each measured state its frequency of occurence
        meta_data = {"n_samples": str(n_samples)}
        if result_samples:
            meta_data["n_qubits"] = str(len(list(result_samples.keys())[0]))

        myqlm_result = Result(meta_data=meta_data)
        for state, counts in result_samples.items():
            if aggregate_samples:
                myqlm_result.add_sample(int(state, 2), probability=counts / n_samples)
                continue
            for _ in range(counts):
                myqlm_result.add_sample(int(state, 2), probability=1 / n_samples)

        myqlm_result.wrap_samples(
            [QRegister(0, length=int(meta_data.get("n_qubits", 0)))]
        )
        return myqlm_result

    @staticmethod
    def convert_result_to_samples(myqlm_result: Result) -> Counter:
        """Converts a MyQLM Result into the output of a sampling.

        To convert a MyQLM Result into the output of a sampling, the number of samples
        performed and the number of qubits need to be precised in the meta_data under
        the keys "n_samples" and "n_qubits" respectively. This is automatically filled
        when the MyQLM Result were obtained from `convert_samples_to_result`.

        Args:
            myqlm_result: A Result instance having information about the measurement
                outcome in raw_data and about the number of samples and number of
                qubits in meta_data.

        Returns:
            A dictionary of strings describing the measured states
                and their respective counts.
        """
        # Associates to each measured state its frequency of occurence
        samples: dict[str, int] = defaultdict(int)
        if not myqlm_result.raw_data:
            return Counter(samples)
        # raw_data is not empty so n_samples and n_qubits must be defined
        if not (
            isinstance(myqlm_result.meta_data, dict)
            and "n_samples" in myqlm_result.meta_data
            and "n_qubits" in myqlm_result.meta_data
        ):
            raise ValueError(
                "Meta data must be a dictionary with n_samples and n_qubits defined."
            )
        try:
            n_qubits = int(myqlm_result.meta_data["n_qubits"])
        except (ValueError, TypeError) as e:
            raise type(e)("n_qubits must be castable to an integer.")
        try:
            n_samples = int(myqlm_result.meta_data["n_samples"])
            if n_samples <= 0:
                raise ValueError
        except (ValueError, TypeError) as e:
            raise type(e)(
                "n_samples must be castable to an integer strictly greater than 0."
            )

        for sample in myqlm_result.raw_data:
            if len(sample.state.bitstring) > n_qubits:
                raise ValueError(
                    f"State {sample.state} is incompatible with number of qubits"
                    f" declared {n_qubits}."
                )
            # Going from counts to probability = counts/n_samples back to counts
            # causes issues with float. We assume the obtained counts is close to
            # an integer
            samples[sample.state.bitstring.zfill(n_qubits)] += round(
                sample.probability * n_samples
            )
        return Counter(samples)

    def submit(self, batch: Batch, **kwargs: Any) -> BatchResult:
        """Executes a batch of jobs and returns the corresponding list of Results.

        If the qpu attribute is None, pulser_simulation is used to simulate the Pulser
        Sequence associated with each MyQLM Job in the batch. If the `qpu` attribute
        is not None, all its arguments can be passed as keyword arguments.

        Args:
            batch: a batch of jobs. If a single job is provided, the job is embedded
                into a Batch, executed, and the first result is returned.

        Keyword Args:
            meta_data: A dictionary to override the meta_data of the batch.
            mem_usage: If qpu is a `QLMaaSQPU`, override default memory usage (in MB)
                calculation.
            core_usage: If qpu is a `QLMaaSQPU`, override default core usage
                calculation.
            nb_nodes: If qpu is a `QLMaaSQPU`, override default number of nodes.

        Returns:
            A batch result.
        """
        if self.qpu is None:
            return super().submit(batch, **kwargs)
        return self.qpu.submit(batch, **kwargs)

    def submit_job(self, job: Job) -> Result:
        """Simulate a MyQLM Job using pulser_simulation.

        If no QPU has been provided, simulation of the Pulser Sequence associated with
        the MyQLM Job is performed using pulser_simulation. Default number of shots is
        `pulser_myqlm.devices.DEFAULT_NUMBER_OF_SHOTS`. Pulser Sequence should be
        provided in job.schedule._other. This attribute must be a string in bytes
        containing a dict serialized with json.dumps. The dict must contain at least one
        key: "abstr_seq".

        Args:
            job: the MyQLM Job to execute.

        Returns:
            A MyQLM Result.
        """
        if self.qpu is not None:
            raise ValueError(
                "`submit_job` must not be used if the qpu attribute is defined,"
                " use the `submit` method instead."
            )
        if job.schedule is None:
            raise QPUException(
                ErrorType.NOT_SIMULATABLE,
                message="IsingAQPU can only execute a schedule job.",
            )
        other_dict = deserialize_other(job.schedule._other)
        modulation = other_dict.get("modulation", False)
        return self.convert_samples_to_result(
            simulate_seq(other_dict["seq"], modulation, job.nbshots), job.aggregate_data
        )

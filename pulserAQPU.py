# Copyright 2020 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
from pulser import Pulse, Register, Sequence
from pulser.channels import Rydberg
from pulser.devices._device_datacls import BaseDevice
from pulser.devices.interaction_coefficients import c6_dict
from pulser.waveforms import CustomWaveform, Waveform
from pulser_simulation import Simulation
from qat.core import Observable, Result, Term
from qat.core.qpu import QPUHandler
from qat.core.variables import ArithExpression, angle, cos, sin, sqrt


def concatenate_times_list(times: list) -> list:
    """Concatenate segments having the same value in a Times.times list.

    Args:
        times: a Times.times list whose segments should be eventually fusioned.

    Returns:
        a Times.times list having some segments combined.
    """
    concatenated_times = []
    n = len(times)
    constant = times[0]
    for i in range(1, n):
        if times[i][2] == constant[2]:
            constant[1] = times[i][1]
        else:
            concatenated_times.append(constant)
            constant = times[i]
    concatenated_times.append(constant)
    return concatenated_times


class Times:
    r"""Class to represent a function over a segment [t0; tmax].

    A class to evaluate the heaviside functions in an expression between two
    times t0 and tmax. The initial and final times of the heaviside function
    define sub-segments between t0 and tmax. The result of this evaluation is
    given in a times list.

    Attributes:
        t0: The starting time of the segment.
        tmax: The ending time of the segment.
        expression: The function to evaluate.
        times: A list composed of segments during which no heaviside function
            is called. These segments are represented by their initial and
            final times, and an expression in which the heaviside function is
            evaluated over this segment.
    """
    t0: int
    tmax: int
    expression: int | float | ArithExpression
    times: list[list[int, int, int | float | ArithExpression]]

    def __init__(self, t0, tmax, expression=None, times=[]) -> None:
        self.t0 = t0
        self.tmax = tmax
        self.expression = expression
        if len(times) == 0:
            if expression is None:
                self.times = []
            else:
                self.times = self.extract_times(t0, tmax, expression)
        else:
            if times[0][0] != self.t0:
                raise ValueError("t0, first element in list should be equal")
            if times[-1][1] != self.tmax:
                raise ValueError("tmax, last time in list should be equal")
            self.times = list(times)

    def __repr__(self) -> str:
        return str(self.times)

    def __str__(self) -> str:
        return str(self.times)

    def totuple(self) -> tuple[tuple[int, int, int | float | ArithExpression]]:
        return tuple([tuple(time) for time in self.times])

    @property
    def length_times(self) -> int:
        return len(self.times)

    @property
    def values(self) -> list[int | float | ArithExpression]:
        return [time[2] for time in self.times]

    @property
    def segments(self) -> list[tuple[int, int]]:
        return [(time[0], time[1]) for time in self.times]

    def extract_times_rec(self, ti: int, tf: int, expression: ArithExpression):
        """Extract a sequence from an expression.

        Work recursively to provide an evaluation of the expression
        from ti to tf.

        Args:
            - ti: minimum time at which to evaluate.
            - tf: maximum time at which to evaluate.
            - expression: expression to evaluate.

        Returns:
            List of tuples containing times and value during which
            the expression is constant.
        """
        if isinstance(expression, ArithExpression):
            heavisideIndex = expression.to_thrift().find("heaviside")
            if heavisideIndex != -1:
                extract_heaviside = expression.to_thrift()[
                    heavisideIndex:
                    ].split()
                try:
                    t1 = int(extract_heaviside[2])  # in ns
                    t2 = int(extract_heaviside[3])  # in ns
                except:
                    raise ValueError("""
                    All times should be defined as integers (in ns)
                    """)
                new_ti = min(t1, t2)
                new_tf = max(t1, t2)
                expression_on = ArithExpression.from_string(
                    expression.to_thrift()[:heavisideIndex]
                    + "1 "
                    + " ".join(extract_heaviside[4:])
                )
                expression_off = ArithExpression.from_string(
                    expression.to_thrift()[:heavisideIndex]
                    + "0 "
                    + " ".join(extract_heaviside[4:])
                )
                if new_ti > tf or new_tf <= ti:
                    return self.extract_times_rec(ti, tf, expression_off)
                elif new_ti <= ti and new_tf >= tf:
                    return self.extract_times_rec(ti, tf, expression_on)
                elif new_ti > ti and new_tf >= tf:
                    return self.extract_times_rec(
                        ti, new_ti - 1, expression_off
                    ) + self.extract_times_rec(new_ti, tf, expression_on)
                elif new_ti <= ti and new_tf < tf:
                    return self.extract_times_rec(
                        ti, new_tf - 1, expression_on
                    ) + self.extract_times_rec(new_tf, tf, expression_off)
                elif new_ti > ti and new_tf < tf:
                    return (
                        self.extract_times_rec(ti, new_ti - 1, expression_off)
                        + self.extract_times_rec(
                            new_ti,
                            new_tf - 1,
                            expression_on,
                            )
                        + self.extract_times_rec(new_tf, tf, expression_off)
                    )
        return [[ti, tf, expression]]

    def extract_times(self, ti: int, tf: int, expression: ArithExpression):
        """Output a concatenated times list from an expression."""
        return concatenate_times_list(
            self.extract_times_rec(
                ti,
                tf,
                expression,
                )
            )

    def multiply_by_expression(self, new_expression):
        """Multiply the times list by a new expression."""
        product = []
        for times_coeff in self.times:
            product += self.extract_times(
                times_coeff[0], times_coeff[1], times_coeff[2] * new_expression
            )
        return product


def sum_times(times_list: list[Times]) -> Times:
    """Sum a list of Times instances, output a Times instance."""
    ntimes = len(times_list)
    if ntimes == 0:
        raise ValueError("""
        At least one Times element should be provided in a list
        """)
    if len(set([times_list[i].t0 for i in range(ntimes)])) != 1:
        raise ValueError("""
        All time lists should start at the same time
        """)
    if len(set([times_list[i].tmax for i in range(ntimes)])) != 1:
        raise ValueError("""
        All time lists should end at the same time
        """)
    t0 = times_list[0].t0
    tmax = times_list[0].tmax
    index_in_times = np.zeros(ntimes, dtype=int)
    summed_times = []
    time = t0
    while time <= tmax:
        final_times = [
            times_list[i].times[index_in_times[i]][1]
            for i in range(ntimes)
            ]
        final_time = min(final_times)
        summed_times.append(
            [
                time,
                final_time,
                sum([
                    times_list[i].times[index_in_times[i]][2]
                    for i in range(ntimes)
                    ]),
            ]
        )
        for i in range(ntimes):
            if final_times[i] == final_time:
                index_in_times[i] += 1
        time = final_time + 1
    return Times(t0, tmax, times=summed_times)


def synch_times(times_list: list[Times]) -> list[Times]:
    """Synchronize a list of Times instances.

    Reshape the times list of the Times instances such that
    the times at which they are evaluated are all the same.
    """
    ntimes = len(times_list)
    if len(set([times_list[i].t0 for i in range(ntimes)])) != 1:
        raise ValueError("All time lists should start at the same time")
    if len(set([times_list[i].tmax for i in range(ntimes)])) != 1:
        raise ValueError("All time lists should end at the same time")
    t0 = times_list[0].t0
    tmax = times_list[0].tmax
    index_in_times = np.zeros(ntimes, dtype=int)
    synched_times = [[] for _ in range(ntimes)]
    time = t0
    while time <= tmax:
        final_times = [
            times_list[i].times[index_in_times[i]][1]
            for i in range(ntimes)
            ]
        final_time = min(final_times)
        for i in range(ntimes):
            synched_times[i].append(
                [time, final_time, times_list[i].times[index_in_times[i]][2]]
            )
            if final_times[i] == final_time:
                index_in_times[i] += 1
        time = final_time + 1

    return [Times(t0, tmax, times=synched_times[i]) for i in range(ntimes)]


class PasqalAQPU(QPUHandler):
    r"""Base class of a Pasqal Analog Quantum Processing Unit.

    A PasqalAQPU is defined by a Device and a Register.
    The Observable given in the job are in :math:`\hbar rad/\mu s`

    Attributes:
        device: The device used for the computations
        register: The register used
    """

    device: BaseDevice
    register: Register
    nbqubits: int
    schedule: Sequence | None = None

    def __init__(self, device: BaseDevice, register: Register) -> None:
        super().__init__()
        self.device = device
        self.register = register
        self.nbqubits = len(self.register._ids)

    @property
    def distances(self):
        r"""Distances between each qubits (in :math:`\mu m`)."""
        positions = self.register._to_abstract_repr()
        distance_matrix = np.zeros((self.nbqubits, self.nbqubits))
        for i in range(self.nbqubits):
            for j in range(self.nbqubits):
                distance_matrix[i][j] += np.sqrt(
                    (positions[i]["x"] - positions[j]["x"]) ** 2
                    + (positions[i]["y"] - positions[j]["y"]) ** 2
                )
        return distance_matrix

    @property
    def interactions(self):
        r"""Interactions between the qubits (in :math:`rad/\mu s`)."""
        interactions = np.zeros((self.nbqubits, self.nbqubits))
        for i in range(self.nbqubits):
            for j in range(self.nbqubits):
                if self.distances[i][j] == 0:
                    interactions[i][j] = 0
                else:
                    interactions[i][j] += (
                        c6_dict[
                            self.device.rydberg_level
                            ] / self.distances[i][j] ** 6
                    )
        return interactions

    def extract_job_params(self, job):
        """Extract all the parameters of the job."""
        # Only the time variable should appear in the next computations
        self.parameter_map = (
            job.parameter_map
        )  # a dict of parameters given in entry: value
        if not job.circuit:
            self.circuit = job.circuit
        else:
            self.circuit = job.circuit(
                **self.parameter_map
            )  # evaluate the circuit with the parameters
        if not job.schedule:
            self.schedule = job.schedule
        else:
            self.schedule = job.schedule(
                **self.parameter_map
            )  # evaluate the sequence with the parameters
        # Other parameters
        self.jobtype = job.type
        self.jobObservable = job.observable
        self.jobqubits = job.qubits
        self.jobnbshots = job.nbshots
        self.jobaggregate_data = job.aggregate_data
        self.jobamp_threshoold = job.amp_threshold
        self.jobpsi_0 = job.psi_0_str
        self.jobvariables = job.get_variables()

    def extract_schedule_params(self):
        """Extract all the parameters of the schedule.

        Returns:
            A list containing a dictionary with the initial and final
            times, and the coefficents in front of each Pauli operator
            for each observable in the schedule
        """
        # Extract main schedule parameters and perform checks
        self.schedulenbqbits = self.schedule.nbqbits
        if self.schedulenbqbits > self.nbqubits:
            raise ValueError("More qubits in Schedule than in Register")
        # Check that no other variables are present than the time variable
        self.scheduleTimeVariable = self.schedule.tname
        self.scheduleVariables = self.schedule.get_variables()
        if self.scheduleTimeVariable[0] in self.scheduleVariables:
            self.scheduleVariables.remove(self.scheduleTimeVariable[0])
        if len(self.scheduleVariables) > 0:
            raise ValueError(
                "Variables" + str(self.scheduleVariables) + "not defined"
                )

        if self.schedule._tmax.int_p is None:
            raise ValueError("tmax should be integer (in ns)")
        else:
            self.tmax = self.schedule._tmax.get_value()
        self.drive_coeffs = self.schedule.drive_coeffs
        self.nsequences = len(self.schedule.drive_obs)

        sequences = []
        for i in range(self.nsequences):
            obs = self.schedule.drive_obs[i]
            parameters = {}
            # evaluate the heaviside functions expressed
            # in the coefficient multiplying hamiltonian number i
            # extract segments during which the heavisides are constant
            pulses = Times(
                0,
                self.tmax,
                expression=self.drive_coeffs[i].get_value())
            # extract the constant coeff as a step function
            parameters["identity"] = Times(
                0,
                self.tmax,
                times=pulses.multiply_by_expression(
                    obs._constant_coeff.get_value()
                    ),
            )
            # extract the coefficient in front of each operator
            # as a Times instance
            for term in obs._terms:
                parameters[(term.op, tuple(term.qbits))] = Times(
                    0,
                    self.tmax,
                    times=pulses.multiply_by_expression(
                        term._coeff.get_value()
                        ),
                )
            sequences.append(parameters)
        return sequences

    def get_sequence(self, sequences):
        """Sum a list of sequence and makes a sequence."""
        sequence = {}
        all_keys = []
        for i in range(self.nsequences):
            all_keys += list(sequences[i].keys())
        keys = list(set(all_keys))
        for key in keys:
            sequence[key] = Times(
                0,
                self.tmax,
                times=concatenate_times_list(
                    sum_times(
                        [
                            sequences[i][key]
                            for i in range(self.nsequences)
                            if key in sequences[i].keys()
                        ]
                    ).times
                ),
            )
        return sequence

    def get_synchronized_sequence(self, sequence):
        """Synchronize all the Times instances in a sequence."""
        keys = list(sequence.keys())
        synchronized_sequence = {}
        synchronized_list = synch_times([sequence[key] for key in keys])
        for i in range(len(keys)):
            synchronized_sequence[keys[i]] = synchronized_list[i]
        return synchronized_sequence

    def extract_gateset_terms(self, sequence, gateset, excludeothers=True):
        """Extract the Times instances of operators in a gateset."""
        ising_sequence = {gate: {} for gate in gateset}
        for key in sequence.keys():
            if key == "identity":
                ising_sequence[key] = sequence[key]
            else:
                if key[0] in gateset:
                    ising_sequence[key[0]][key[1]] = sequence[(key[0], key[1])]
                elif excludeothers:
                    raise ValueError(
                        "Only Ising terms are accepted:" + str(gateset)
                        )
        return ising_sequence

    def evaluateTimes(self, time: list):
        """Converts a segment of a Times instance into a Waveform.

        A segment is [ti, tf, expression].
        If the expression is time dependent, then a waveform is returned.
        Otherwise the expression is constant along time, this value is returned
        """
        if isinstance(time[2], ArithExpression):
            if self.scheduleTimeVariable in time[2].get_variables():
                timearray = [
                    time[2](**{self.scheduleTimeVariable: t_value})
                    for t_value in range(time[0], time[1] + 1, 1)
                ]
                if len(set(timearray)) > 1:
                    return CustomWaveform(np.array(timearray))
                else:
                    return timearray[0]
        return time[2]

    def convertToCompBasis(self, statename):
        """Converts a string describing a state with 0 and 1 into a number."""
        n = len(statename)
        statevalue = 0
        for i in range(n):
            statevalue += int(statename[i]) * 2 ** (n - 1 - i)
        return statevalue

    def convertPulserResult(self, pulserResult, N_samples, asdict):
        """Converts a Pulser result into a myqlm result or a dict."""
        myqlmResult = Result(nbqbits=self.nbqubits)
        if self.jobtype == 0:
            result_dict = pulserResult.sample_final_state(N_samples=N_samples)
            if asdict:
                return result_dict
            for k, v in result_dict.items():
                state = self.convertToCompBasis(k)
                myqlmResult.add_sample(state, probability=float(v) / N_samples)
            return myqlmResult
        elif self.jobtype == 1:
            raise ValueError("""
            OBS measurement not implemented on PasqalAQPU
            """)
        else:
            raise ValueError("""
            Job type can only be "OBS" or "SAMPLE".
            """)

    def _init_submit_job(self, job):
        self.extract_job_params(job)
        myqlmSequences = self.extract_schedule_params()
        myqlmSequence = self.get_sequence(myqlmSequences)
        myqlmSynchronizedSequence = self.get_synchronized_sequence(
            myqlmSequence
            )
        return myqlmSynchronizedSequence

    def _submit_sequence(
        self,
        sequence,
        sampling_rate,
        config,
        evaluation_times,
        with_modulation,
        N_samples,
        asdict,
    ):
        self.sim = Simulation(
            sequence, sampling_rate, config, evaluation_times, with_modulation
        )
        pulserResult = self.sim.run(progress_bar=True)
        outputResult = self.convertPulserResult(
            pulserResult,
            N_samples,
            asdict,
            )
        return outputResult


class FresnelAQPU(PasqalAQPU):
    r"""Fresnel Analog Quantum Processing Unit.

    Device and register should respect a certain set of rules:
        - Device needs at least a Rydberg channel with global addressing
        - Pre-calibrated layouts of the device
        should be among the one implemented on Fresnel
        - Can only implement Ising Hamiltonians

    Args:
        device: A device having a Rydberg.Global channel.
        register: A register defining the interactions between the atoms.
    """
    def __init__(self, device: BaseDevice, register: Register) -> None:
        super().__init__(device=device, register=register)
        self.nbqubits = len(self.register._ids)
        self.check_channels(self.device)

    def check_channels(self, device: BaseDevice) -> None:
        """Check that the device has a Rydberg.Global channel."""
        has_rydberg_global = False
        for channel_name, channel_type in device.channels.items():
            if (
                isinstance(channel_type, Rydberg)
                and channel_type.addressing == "Global"
            ):
                has_rydberg_global = True
                self.ising_channel = channel_name
        if not has_rydberg_global:
            raise ValueError(
                """
                Fresnel AQPU: the device should at least have
                a Rydberg channel with Global addressing
                """
            )

    def ising_hamiltonian(
        self,
        nqubits: int,
        Omega_t: Observable,
        delta_t: Observable,
        phi: float
    ):
        """Defines a Ising Hamiltonian using the register of the device.

        Args:
            blockade_radius: The Rydberg blockade radius, in µm.

        Returns:
            The maximum rabi frequency value, in rad/µs.
        """
        hamiltonian = 0
        for i in range(nqubits):
            hamiltonian += (
                Omega_t
                / 2
                * (
                    cos(phi) * Observable(
                        nqubits,
                        pauli_terms=[Term(1, "X", [i])]
                        )
                    - sin(phi) * Observable(
                        nqubits,
                        pauli_terms=[Term(1, "Y", [i])]
                        )
                )
            )
            hamiltonian -= (
                delta_t
                / 4
                * Observable(
                    nqubits, pauli_terms=[
                        Term(1, "I", [i]),
                        Term(-1, "Z", [i])
                        ]
                )
            )
            for j in range(i):
                hamiltonian += (
                    self.interactions[i][j]
                    / 4
                    * Observable(
                        nqubits,
                        pauli_terms=[
                            Term(1, "II", [i, j]),
                            Term(-1, "IZ", [i, j]),
                            Term(-1, "ZI", [i, j]),
                            Term(1, "ZZ", [i, j]),
                        ],
                    )
                )
        return hamiltonian

    def check_interactions(self, isingSequence):
        """Check the correspondence interactions-ZZ coefficients."""
        ZZsequence = isingSequence["ZZ"]
        for qubitid in ZZsequence:
            ZZcoeff = list(set(ZZsequence[qubitid].values))
            if 0 in ZZcoeff and len(ZZcoeff) > 1:
                ZZcoeff.pop(ZZcoeff.index(0))
            if len(ZZcoeff) > 1:
                raise ValueError(
                    "ZZ coefficients should be constant or 0 along time"
                    )
            if isinstance(ZZcoeff[0], ArithExpression):
                raise ValueError(
                    "ZZ coefficients should be time-independent"
                    )
            if (
                4 * ZZcoeff[0]
                != self.interactions[max(qubitid[0], qubitid[1])][
                    min(qubitid[0], qubitid[1])
                ]
            ):
                raise ValueError(
                    """
                    ZZ coefficients should match with defined Register:
                    use qpu.interactions to define them
                    """
                )

    def extract_omegas_phases(self, isingSequence: dict[str:dict]):
        """Returns Times instances for the Rabi frequency and phase."""
        X_times = list(
            set(
                [
                    isingSequenceX.totuple()
                    for isingSequenceX in isingSequence["X"].values()
                ]
            )
        )  # Warning: set of my own class ?
        Y_times = list(
            set(
                [
                    isingSequenceY.totuple()
                    for isingSequenceY in isingSequence["Y"].values()
                ]
            )
        )
        nxcoeff = len(X_times)
        nycoeff = len(Y_times)
        if nxcoeff > 1 or nycoeff > 1:
            raise ValueError("Omega should be the same on each qubit")
        if nxcoeff == 0:
            X_times = Times(0, self.tmax, X_times)
        if nycoeff == 0:
            Y_times = Times(0, self.tmax, Y_times)
        omega_list = []
        phase_list = []
        for i in range(len(X_times[0])):
            r = X_times[0][i][2] + 1j * Y_times[0][i][2]
            phase = -1 * angle(r)
            if isinstance(phase, ArithExpression):
                phase_array = [
                    phase(t=ti)
                    for ti in range(X_times[0][i][0], X_times[0][i][1] + 1)
                ]
                phase = np.mean(phase_array)
            omega = sqrt(X_times[0][i][2] ** 2 + Y_times[0][i][2] ** 2)
            omega_list.append([X_times[0][i][0], X_times[0][i][1], omega])
            phase_list.append([X_times[0][i][0], X_times[0][i][1], phase])

        return Times(0, self.tmax, times=omega_list), Times(
            0, self.tmax, times=phase_list
        )

    def extract_deltas(self, isingSequence):
        """Returns Times instance of the detuning."""
        # Coeff in front of I
        # = -\delta / 4 + \sum_{i!=j}U_{i, j}/8
        constant_from_U = np.sum(self.interactions) / 8
        identity_sequence = isingSequence["identity"]
        deltas_list = []
        for identity_time in identity_sequence.times:
            deltas_list.append(
                [
                    identity_time[0],
                    identity_time[1],
                    -4 * (identity_time[2] - constant_from_U),
                ]
            )
        return Times(0, self.tmax, times=deltas_list)

    def build_ising_sequence(self, omegas, deltas, phases):
        """Build a pulse sequence on the Rydberg.Global channel."""
        for i in range(omegas.length_times):

            duration = omegas.times[i][1] - omegas.times[i][0] + 1

            omega = self.evaluateTimes(omegas.times[i])
            isTimeDependentOmega = isinstance(omega, Waveform)

            delta = self.evaluateTimes(deltas.times[i])
            isTimeDependentDelta = isinstance(delta, Waveform)

            phase = phases.times[i][2]
            assert not isinstance(phase, ArithExpression)

            if not isTimeDependentOmega and not isTimeDependentDelta:
                # If all parameters are constant
                # Add a constant pulse
                self.sequence.add(
                    Pulse.ConstantPulse(duration, omega, delta, phase),
                    channel="ising_ch",
                )
            elif not isTimeDependentOmega:
                # In this case we could like to send an eom pulse
                self.sequence.add(
                    Pulse.ConstantAmplitude(duration, omega, delta, phase),
                    channel="ising_ch",
                )
            elif not isTimeDependentDelta:
                self.sequence.add(
                    Pulse.ConstantDetuning(duration, omega, delta, phase),
                    channel="ising_ch",
                )  # Does not work
            else:
                self.sequence.add(
                    Pulse(omega, delta, phase),
                    channel="ising_ch",
                    )

    def submit_job(
        self,
        job,
        toprint=True,
        todraw=True,
        sampling_rate=1.0,
        config=None,
        evaluation_times="Full",
        with_modulation=False,
        N_samples=1000,
        asdict=False,
    ):
        """Runs a job on Pulser."""
        if list(set(job.psi_0))[0] != "0":
            raise ValueError(
                "Fresnel: qubit should be initialized in |0> state"
                )

        myqlmSynchSequence = self._init_submit_job(job)

        # Extract Omega(t), delta(t) for each sequence
        ising_gateset = ["X", "Y", "Z", "ZZ"]
        isingSynchSequence = self.extract_gateset_terms(
            myqlmSynchSequence, ising_gateset, excludeothers=True
        )
        self.check_interactions(isingSynchSequence)
        omegas, phases = self.extract_omegas_phases(isingSynchSequence)
        deltas = self.extract_deltas(isingSynchSequence)
        # Build Ising sequence
        self.sequence = Sequence(self.register, self.device)
        self.sequence.declare_channel("ising_ch", self.ising_channel)

        self.build_ising_sequence(omegas, deltas, phases)
        if toprint:
            print(self.sequence)
        if todraw:
            self.sequence.draw()
        # Configure slm for self.psi_0 initialization
        # Configure measurements from self.measObservable
        return self._submit_sequence(
            self.sequence,
            sampling_rate,
            config,
            evaluation_times,
            with_modulation,
            N_samples,
            asdict,
        )

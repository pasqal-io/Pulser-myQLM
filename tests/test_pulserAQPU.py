from __future__ import annotations

import json
from collections import Counter
from threading import Thread
from time import sleep
from unittest import mock

import numpy as np
import pytest
from pulser import Pulse, Sequence
from pulser.channels import Raman, Rydberg
from pulser.devices import MockDevice, VirtualDevice
from pulser.devices.interaction_coefficients import c6_dict
from pulser.register import Register
from pulser.waveforms import CustomWaveform
from pulser_simulation import QutipEmulator
from qat.comm.exceptions.ttypes import QPUException
from qat.core import Job, Result, Sample, Schedule
from qat.core.qpu import QPUHandler
from qat.core.variables import ArithExpression, Symbol, cos, sin
from qat.lang.AQASM import CCNOT, Program
from qat.qpus import PyLinalg, RemoteQPU
from thrift.transport.TTransport import TTransportException

from pulser_myqlm.myqlmtools import are_equivalent_schedules, sample_schedule
from pulser_myqlm.pulserAQPU import TEMP_DEVICE, FresnelQPU, IsingAQPU


def Pmod(a: float, b: float) -> float:
    """Returns rest of euclidian division of a by b."""
    return a % b


mod = Symbol(token="%", evaluator=Pmod, arity=2)


@pytest.mark.parametrize("mod_value", [(0, 0), (1, 1), (10, 0), (15, 5)])
def test_mod(mod_value, u_variable):
    a = mod_value[0]
    result = mod_value[1]
    assert Pmod(a, 10) == result
    assert mod(a, 10) == result
    mod_expr = mod(u_variable, 10)
    assert isinstance(mod_expr, ArithExpression)
    assert mod_expr.get_variables() == ["u"]
    assert mod_expr.to_thrift() == "% u 10"
    assert mod_expr(u=a) == result


def test_nbqubits(test_ising_qpu):
    assert test_ising_qpu.nbqubits == 3


def test_distances(test_ising_qpu):
    dist_tl = np.array(
        [
            [0, 0, 0],
            [5, 0, 0],
            [5, 5, 0],
        ]
    )
    assert np.all(test_ising_qpu.distances == dist_tl + dist_tl.T)


def test_ising_init(test_ising_qpu):
    with pytest.raises(TypeError, match="The provided device must be"):
        IsingAQPU(12, test_ising_qpu.register)
    with pytest.raises(TypeError, match="The provided register must be"):
        IsingAQPU(test_ising_qpu.device, 12)
    with pytest.raises(TypeError, match="The provided qpu must be"):
        IsingAQPU(test_ising_qpu.device, test_ising_qpu.register, 12)
    assert test_ising_qpu.channel == "rydberg_global"
    # Test the Ising AQPU with no rydberg global channel
    device = VirtualDevice(
        name="VirtDevice",
        dimensions=2,
        channel_objects=(
            Rydberg(
                "Local",
                max_abs_detuning=None,
                max_amp=None,
                min_retarget_interval=1,
                fixed_retarget_t=1,
            ),
            Raman("Global", max_abs_detuning=None, max_amp=None),
        ),
        rydberg_level=60,
    )
    with pytest.raises(
        ValueError,
        match="""
                Ising AQPU: the device should at least have
                a Rydberg channel with Global addressing.
                """,
    ):
        IsingAQPU(device=device, register=test_ising_qpu.register)


def test_c6_interactions(test_ising_qpu):
    assert np.all(
        np.diagonal(test_ising_qpu.c6_interactions)
        == np.zeros((1, test_ising_qpu.nbqubits))
    )
    int_edge = c6_dict[test_ising_qpu.device.rydberg_level] / 5**6
    int_tl = np.array(
        [
            [0, 0, 0],
            [int_edge, 0, 0],
            [int_edge, int_edge, 0],
        ]
    )
    assert np.all(test_ising_qpu.c6_interactions == int_tl + int_tl.T)


def test_interaction_observables(test_ising_qpu):
    """Test time-independent coeff in front of I, Z and ZZ operators."""
    assert test_ising_qpu.nbqubits == test_ising_qpu.interaction_observables.nbqbits
    # Testing the coefficient in front of each operator
    # Each c6 interaction adds c6_interaction / 4 to coeff in front of I
    assert np.isclose(
        test_ising_qpu.interaction_observables.constant_coeff,
        np.sum(np.tril(test_ising_qpu.c6_interactions) / 4.0),
        rtol=1e-15,
    )
    for term in test_ising_qpu.interaction_observables.terms:
        # Only "Z" or "ZZ" operator
        assert term.op in ["Z", "ZZ"]
        if term.op == "ZZ":
            # c6 int btw qbit i and j adds c6_interaction / 4 to Z_iZ_j coeff
            assert (
                term._coeff.get_value()
                == test_ising_qpu.c6_interactions[term.qbits[0]][term.qbits[1]] / 4.0
            )
        elif term.op == "Z":
            # c6 int btw qbit i and j adds c6_interaction / 4 to coeff in front of Z_i
            assert (
                term._coeff.get_value()
                == np.sum(test_ising_qpu.c6_interactions[term.qbits[0]][:]) / 4.0
            )


@pytest.mark.parametrize(
    "amp, det, phase",
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, np.pi / 2),
        (1, 0, 0.3),
        (1, 0, "u_variable"),
        (0, 1, 0),
        (0, 1, np.pi / 2),
        ("omega_t", "delta_t", 0),
    ],
)
def test_pulse_observables(test_ising_qpu, amp, det, phase, request):
    """Test time-dependent coeffs in front of X, Y, Z when applying a pulse."""

    def phase_test(dict_terms, index):
        """Test the coeffs in front of X and Y operators when phase=0 or pi/2."""
        if index == "X":
            opp_index = "Y"
        else:
            opp_index = "X"
        assert len(dict_terms[opp_index]) == 0
        assert len(dict_terms[index]) == test_ising_qpu.nbqubits
        assert len(set(dict_terms[index].values())) == 1
        phase = (-1) ** (index == "Y") * amp / 2.0
        assert set(dict_terms[index].values()) == {
            phase if not isinstance(phase, ArithExpression) else phase.to_thrift()
        }

    amp, det, phase = (
        (
            request.getfixturevalue(pulse_attr)
            if isinstance(pulse_attr, str)
            else pulse_attr
        )
        for pulse_attr in (amp, det, phase)
    )

    obs = test_ising_qpu.pulse_observables(amp, det, phase)
    assert test_ising_qpu.nbqubits == obs.nbqbits
    # Coefficient in front of I is zero
    assert obs.constant_coeff == 0.0
    # Get the coefficients in front of X, Y and Z operators
    dict_terms = {"X": {}, "Y": {}, "Z": {}}
    for term in obs.terms:
        assert term.op in dict_terms.keys()
        coeff = term._coeff.get_value()
        if isinstance(coeff, ArithExpression):
            dict_terms[term.op][term.qbits[0]] = coeff.to_thrift()
            continue
        dict_terms[term.op][term.qbits[0]] = coeff

    # X and Y coefficients are associated to amp and phase
    if amp == 0:
        # No X and Y observables if amp is zero
        assert len(dict_terms["X"]) == 0
        assert len(dict_terms["Y"]) == 0
    elif mod(phase, np.pi) == 0:
        # 0 Y observables, nqubits X observables
        phase_test(dict_terms, "X")
    elif mod(phase, np.pi) == np.pi / 2:
        # 0 X observables, nqubits Y observables
        phase_test(dict_terms, "Y")
    else:
        # nqubits X observables and Y observables
        assert (
            len(dict_terms["X"]) == len(dict_terms["Y"])
            and len(dict_terms["X"]) == test_ising_qpu.nbqubits
        )
        assert (
            len(set(dict_terms["X"].values())) == 1
            and len(set(dict_terms["Y"].values())) == 1
        )
        x_coeff = 0.5 * cos(phase) * amp
        y_coeff = -0.5 * sin(phase) * amp
        assert set(dict_terms["X"].values()) == {
            x_coeff if not isinstance(x_coeff, ArithExpression) else x_coeff.to_thrift()
        }
        assert set(dict_terms["Y"].values()) == {
            y_coeff if not isinstance(y_coeff, ArithExpression) else y_coeff.to_thrift()
        }

    # Z coefficients are associated to det
    if det == 0:
        assert len(dict_terms["Z"]) == 0
    else:
        assert len(dict_terms["Z"]) == test_ising_qpu.nbqubits
        assert len(set(dict_terms["Z"].values())) == 1
        z_coeff = -det / 2.0
        assert set(dict_terms["Z"].values()) == {
            z_coeff if not isinstance(z_coeff, ArithExpression) else z_coeff.to_thrift()
        }


@pytest.mark.parametrize(
    "amp, det, phase",
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, np.pi / 2),
        (1, 0, 0.3),
        (1, 0, "u_variable"),
        (0, 1, 0),
        (0, 1, np.pi / 2),
        ("omega_t", "delta_t", 0),
    ],
)
def test_hamiltonian(test_ising_qpu, amp, det, phase, request):
    """Test Ising Hamiltonian generated by pulses."""
    amp, det, phase = (
        (
            request.getfixturevalue(pulse_attr)
            if isinstance(pulse_attr, str)
            else pulse_attr
        )
        for pulse_attr in (amp, det, phase)
    )
    ising_ham = test_ising_qpu.hamiltonian(amp, det, phase)
    if (amp, det, phase) == (0, 0, 0):
        assert ising_ham == test_ising_qpu.interaction_observables
    else:
        dict_terms = {"X": {}, "Y": {}, "Z": {}, "ZZ": {}}
        for term in ising_ham.terms:
            assert term.op in dict_terms.keys()
            coeff = term._coeff.get_value()
            dict_terms[term.op][tuple(term.qbits)] = (
                coeff.to_thrift() if isinstance(coeff, ArithExpression) else coeff
            )

        dict_ising_int = {"Z": {}, "ZZ": {}}
        for term in test_ising_qpu.interaction_observables.terms:
            coeff = term._coeff.get_value()
            dict_ising_int[term.op][tuple(term.qbits)] = (
                coeff.to_thrift() if isinstance(coeff, ArithExpression) else coeff
            )

        for qbits, term_coeff in dict_terms["Z"].items():
            assert qbits in dict_ising_int["Z"].keys()
            z_coeff = dict_ising_int["Z"][qbits] + -det / 2.0
            assert (
                term_coeff == z_coeff.to_thrift()
                if isinstance(z_coeff, ArithExpression)
                else z_coeff
            )


@pytest.mark.parametrize("device_type", ["raman", "local"])
def test_convert_init_sequence_to_schedule(test_ising_qpu, device_type):
    """Testing IsingAQPU.convert_sequence_to_schedule."""
    # An empty sequence returns an empty Schedule
    seq = Sequence(test_ising_qpu.register, MockDevice)
    assert Schedule() == IsingAQPU.convert_sequence_to_schedule(seq)
    # Conversion only works for Rydberg Global channel
    # Does not work if a Raman Global channel is declared
    if device_type == "raman":
        seq.declare_channel("ram_glob", "raman_global")
        with pytest.raises(
            TypeError,
            match="Declared channel is not Rydberg.",
        ):
            IsingAQPU.convert_sequence_to_schedule(seq)
    # Does not work if a Local Rydberg channel is declared
    elif device_type == "local":
        seq.declare_channel("ryd_loc", "rydberg_local")
        with pytest.raises(
            TypeError,
            match="Declared channel is not Rydberg.Global.",
        ):
            IsingAQPU.convert_sequence_to_schedule(seq)
    # Does not work if multiple Rydberg Global channels are declared
    seq = Sequence(test_ising_qpu.register, MockDevice)
    seq.declare_channel("ryd_glob", "rydberg_global")
    seq.declare_channel("ryd_glob1", "rydberg_global")
    with pytest.raises(
        ValueError,
        match="More than one channel declared.",
    ):
        IsingAQPU.convert_sequence_to_schedule(seq)


@pytest.fixture
def failing_schedule_seq(test_ising_qpu, omega_t, delta_t):
    """(Schedule, Sequence) who are not equivalent due to MyQLM's get_item."""
    t0 = 16 / 1000  # in µs
    H0 = test_ising_qpu.hamiltonian(omega_t, delta_t, 0)
    t1 = 20 / 1000  # in µs
    H1 = test_ising_qpu.hamiltonian(1, 0, 0)
    t2 = 20 / 1000  # in µs
    H2 = test_ising_qpu.hamiltonian(1, 0, np.pi / 2)

    schedule0 = Schedule(drive=[(1, H0)], tmax=t0)
    schedule1 = Schedule(drive=[(1, H1)], tmax=t1)
    schedule2 = Schedule(drive=[(1, H2)], tmax=t2)
    schedule = schedule0 | schedule1 | schedule2

    # Which is equivalent to having defined pulses using a Sequence
    seq = Sequence(test_ising_qpu.register, test_ising_qpu.device)
    seq.declare_channel("ryd_glob", "rydberg_global")

    seq.add(
        Pulse(
            CustomWaveform([omega_t(t=ti / 1000) for ti in range(int(t0 * 1000))]),
            CustomWaveform(
                [delta_t(t=ti / 1000, u=0) for ti in range(int(t0 * 1000))]
            ),  # no parametrized sequence for the moment
            0,
        ),
        "ryd_glob",
    )
    seq.add(
        Pulse.ConstantPulse(int(t1 * 1000), 1, 0, 0), "ryd_glob", protocol="no-delay"
    )
    seq.add(
        Pulse.ConstantPulse(int(t2 * 1000), 1, 0, np.pi / 2),
        "ryd_glob",
        protocol="no-delay",
    )
    return (schedule, seq)


@pytest.fixture
def schedule_seq(test_ising_qpu, omega_t, delta_t):
    """A tuple of (Schedule, Sequence) who are equivalent."""
    t0 = 16 / 1000  # in µs
    H0 = test_ising_qpu.hamiltonian(omega_t, delta_t, 0)
    t1 = 24 / 1000  # in µs
    H1 = test_ising_qpu.hamiltonian(1, 0, 0)
    t2 = 20 / 1000  # in µs
    H2 = test_ising_qpu.hamiltonian(1, 0, np.pi / 2)

    schedule0 = Schedule(drive=[(1, H0)], tmax=t0)
    schedule1 = Schedule(drive=[(1, H1)], tmax=t1)
    schedule2 = Schedule(drive=[(1, H2)], tmax=t2)
    schedule = schedule0 | schedule1 | schedule2

    # Which is equivalent to having defined pulses using a Sequence
    seq = Sequence(test_ising_qpu.register, test_ising_qpu.device)
    seq.declare_channel("ryd_glob", "rydberg_global")

    seq.add(
        Pulse(
            CustomWaveform([omega_t(t=ti / 1000) for ti in range(int(t0 * 1000))]),
            CustomWaveform(
                [delta_t(t=ti / 1000, u=0) for ti in range(int(t0 * 1000))]
            ),  # no parametrized sequence for the moment
            0,
        ),
        "ryd_glob",
    )
    seq.add(
        Pulse.ConstantPulse(int(t1 * 1000), 1, 0, 0), "ryd_glob", protocol="no-delay"
    )
    seq.add(
        Pulse.ConstantPulse(int(t2 * 1000), 1, 0, np.pi / 2),
        "ryd_glob",
        protocol="no-delay",
    )
    return (schedule, seq)


def test_convert_sequence_to_schedule(schedule_seq):
    """Test conversion of a Sequence in a Schedule using IsingAQPU."""
    schedule, seq = schedule_seq
    schedule_from_seq = IsingAQPU.convert_sequence_to_schedule(seq)
    assert isinstance(schedule_from_seq, Schedule)
    assert are_equivalent_schedules(schedule(u=0), schedule_from_seq)


def test_convert_sequence_with_failing_schedule(failing_schedule_seq):
    """The conversion is correct, but the schedule fails for t=36."""
    schedule, seq = failing_schedule_seq
    schedule_from_seq = IsingAQPU.convert_sequence_to_schedule(seq)
    assert isinstance(schedule_from_seq, Schedule)
    # Schedules are not equivalent
    assert not are_equivalent_schedules(schedule(u=0), schedule_from_seq)
    # Sample the schedules
    sample_sch_from_seq = sample_schedule(schedule_from_seq)
    sample_sch = sample_schedule(schedule(u=0))
    # Extract the value at t=36 ns
    schedule_from_seq_at_36 = sample_sch_from_seq.pop(36)
    schedule_at_36 = sample_sch.pop(36)

    # The schedules are equivalent outside t=36 ns
    assert sample_sch_from_seq == sample_sch
    # In schedule_from_seq, the value at t=36 matches the values of the constant pulse
    # happening after t=36
    assert schedule_from_seq_at_36 == sample_sch_from_seq[-1]
    # In schedule, the value at t=36 matches the values of the constant pulse
    # happening before t=36
    assert schedule_at_36 != sample_sch_from_seq[-1]
    assert schedule_at_36 == sample_sch_from_seq[16]
    # Conclusion: This is an issue with the evaluation of times in MyQLM's get_item


@pytest.mark.parametrize(
    "meta_data, err_mess",
    [
        ({}, "Meta data must be a dictionary."),
        ({"n_qubits": None, "n_samples": 0}, "n_qubits must be castable."),
        (
            {"n_qubits": 1, "n_samples": 0},
            "n_samples must be castable to an integer strictly greater than 0.",
        ),
        (
            {"n_qubits": 1, "n_samples": 1000},
            "State |000> is incompatible with number of qubits declared 1",
        ),
        (
            {"n_qubits": 4, "n_samples": 999},
            "Probability associated with state |000> does not",
        ),
    ],
)
def test_conversion_sampling_result(meta_data, err_mess, schedule_seq, test_ising_qpu):
    """Test the conversion of MyQLM Result into a Counter as in Pulser."""
    np.random.seed(123)
    _, seq = schedule_seq
    sim_result = QutipEmulator.from_sequence(seq, sampling_rate=0.1).run()
    n_samples = 1000
    sim_samples = sim_result.sample_final_state(n_samples)
    sim_samples_dict = {k: v for k, v in sim_samples.items()}
    # Testing the conversion of the pulser samples in a myqlm result
    myqlm_result = test_ising_qpu.convert_samples_to_result(sim_samples)
    myqlm_result_from_dict = test_ising_qpu.convert_samples_to_result(sim_samples_dict)
    assert myqlm_result.meta_data["n_samples"] == str(
        n_samples
    ) and myqlm_result.meta_data["n_qubits"] == str(test_ising_qpu.nbqubits)

    myqlm_samples = {
        sample.state.int: sample.probability for sample in myqlm_result.raw_data
    }
    myqlm_samples_from_dict = {
        sample.state.int: sample.probability
        for sample in myqlm_result_from_dict.raw_data
    }
    assert (
        {int(k, 2): v / n_samples for k, v in sim_samples.items()}
        == myqlm_samples
        == myqlm_samples_from_dict
    )
    # Testing the conversion of a myqlm Result into pulser samples
    # for an empty Result:
    assert test_ising_qpu.convert_result_to_samples(Result()) == Counter()
    # for the sequence above:
    assert test_ising_qpu.convert_result_to_samples(myqlm_result) == sim_samples
    assert (
        test_ising_qpu.convert_result_to_samples(myqlm_result_from_dict) == sim_samples
    )
    # for incorrect meta-data
    myqlm_result.meta_data = meta_data
    with pytest.raises(
        (
            TypeError
            if ("n_qubits" in meta_data and meta_data["n_qubits"] is None)
            else ValueError
        ),
        match=err_mess,
    ):
        test_ising_qpu.convert_result_to_samples(myqlm_result)


@pytest.mark.parametrize("modulation", [False, True])
def test_convert_sequence_to_job(schedule_seq, modulation):
    """Test conversion of a Sequence into a Job."""
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq, modulation=modulation)
    schedule_from_seq = IsingAQPU.convert_sequence_to_schedule(seq, modulation)
    # Schedules obtained from conversion to job and schedule should match
    assert are_equivalent_schedules(schedule_from_seq, job_from_seq.schedule)
    assert job_from_seq.nbshots == 0
    assert schedule_from_seq.to_job().nbshots is None
    assert job_from_seq.schedule._other == schedule_from_seq._other


@pytest.mark.parametrize(
    "other_value, err_mess",
    [
        (None, "job.schedule._other must be a string encoded in bytes."),
        (
            json.dumps({"modulation": True}).encode("utf-8"),
            "An abstract representation of the Sequence",
        ),
        (
            json.dumps({"abstr_seq": "0", "modulation": False}).encode("utf-8"),
            "Failed to deserialize the value",
        ),
    ],
)
def test_job_deserialization(schedule_seq, other_value, err_mess):
    """Test value of Job.schedule._other for the result of a Sequence conversion."""
    schedule, seq = schedule_seq
    aqpu = IsingAQPU.from_sequence(seq)
    job = schedule.to_job()
    job.schedule._other = other_value
    with pytest.raises(ValueError, match=err_mess):
        aqpu.submit(job)
    aqpu.set_qpu(FresnelQPU(None))
    with pytest.raises(
        QPUException, match="Failed at deserializing Job.Schedule._other"
    ):
        aqpu.submit(job)


def deploy_qpu(qpu: QPUHandler, port: int) -> None:
    """Deploys the QPU on a server on a port at IP 127.0.0.1."""
    qpu.serve(port, "127.0.0.1")


def get_remote_qpu(port: int) -> RemoteQPU:
    tries = 0
    while tries < 10:
        try:
            return RemoteQPU(port, "localhost")
        except TTransportException as e:
            tries += 1
            sleep(1)
            error = e
    raise error


def compare_results_raw_data(results1: list, results2: list[tuple]) -> None:
    """Check that two lists of samples (as Result.raw_data) are the same."""
    for i, sample1 in enumerate(results1):
        res_sample1 = (sample1.probability, sample1._state, sample1.state.__str__())
        res_sample2 = (
            results2[i][0].probability,
            results2[i][0]._state,
            results2[i][1],
        )
        assert res_sample1 == res_sample2


port = 1190


@pytest.mark.parametrize("qpu", [None, "local", "remote"])
def test_run_sequence(schedule_seq, qpu):
    """Test simulation of a Sequence using pulser-simulation."""
    np.random.seed(123)
    schedule, seq = schedule_seq
    sim_qpu = None
    # If qpu is None, pulser-simulation in IsingAQPU is used
    if qpu == "local":
        # pulser-simulation in FresnelQPU is used
        sim_qpu = FresnelQPU(None)
        assert sim_qpu.is_operational
        sim_qpu.check_system()
    if qpu == "remote":
        # pulser-simulation in a Remote FresnelQPU is used
        # Deploying a FresnelQPU on a remote server using serve
        server_thread = Thread(target=deploy_qpu, args=(FresnelQPU(None), port))
        server_thread.daemon = True
        server_thread.start()
        # Accessing it with RemoteQPU
        sim_qpu = get_remote_qpu(port)

    aqpu = IsingAQPU.from_sequence(seq, qpu=sim_qpu)
    # IsingQPU and FresnelQPU can only run job with a schedule
    # Defining a job from a circuit instead of a schedule
    prog = Program()
    qbits = prog.qalloc(CCNOT.arity)
    CCNOT(qbits)
    job = prog.to_circ().to_job()
    with pytest.raises(
        QPUException, match="FresnelQPU can only execute a schedule job."
    ):
        aqpu.submit(job)

    # Run job created from a sequence using convert_sequence_to_job
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq, nbshots=1000)
    assert job_from_seq.nbshots == 1000
    result = aqpu.submit(job_from_seq)
    exp_result = [
        (Sample(probability=0.999, state=0), "|000>"),
        (Sample(probability=0.001, state=4), "|100>"),
    ]
    compare_results_raw_data(result.raw_data, exp_result)
    assert IsingAQPU.convert_result_to_samples(result) == {"000": 999, "100": 1}
    # Run job created from a sequence using convert_sequence_to_schedule
    schedule_from_seq = aqpu.convert_sequence_to_schedule(seq)
    job_from_seq = schedule_from_seq.to_job()  # manually defining number of shots
    assert not job_from_seq.nbshots
    result_schedule = aqpu.submit(job_from_seq)
    exp_result_schedule = [
        (Sample(probability=0.9995, state=0), "|000>"),
        (Sample(probability=0.0005, state=1), "|001>"),
    ]
    compare_results_raw_data(result_schedule.raw_data, exp_result_schedule)
    assert IsingAQPU.convert_result_to_samples(result_schedule) == {
        "000": 1999,
        "001": 1,
    }

    # Can simulate Job if Schedule is not equivalent to Sequence
    empty_job = Job()
    empty_schedule = Schedule()
    empty_schedule._other = schedule_from_seq._other
    empty_job.schedule = empty_schedule
    result_empty_sch = aqpu.submit(empty_job)
    exp_result_empty_sch = [
        (Sample(probability=0.999, state=0), "|000>"),
        (Sample(probability=0.0005, state=1), "|001>"),
        (Sample(probability=0.0005, state=4), "|100>"),
    ]
    compare_results_raw_data(result_empty_sch.raw_data, exp_result_empty_sch)
    assert IsingAQPU.convert_result_to_samples(result_empty_sch) == {
        "000": 1998,
        "001": 1,
        "100": 1,
    }

    # Submit_job of IsingAQPU must not be used if qpu is not None
    if qpu is not None:
        with pytest.raises(
            ValueError,
            match="`submit_job` must not be used if the qpu attribute is defined,",
        ):
            aqpu.submit_job(schedule.to_job())

    # Can't simulate a time-dependent job with PyLinalg (a simulator of circuits)
    aqpu.set_qpu(PyLinalg())
    with pytest.raises(TypeError, match="'NoneType' object is not"):
        aqpu.submit(schedule.to_job())


# Whether the current session has access to AnalogQPU
has_analog_qpu = True
try:
    from qlmaas.qpus import AnalogQPU

    myqlm_analog_qpu = AnalogQPU()
except ImportError:
    has_analog_qpu = False


@pytest.mark.skipif(not has_analog_qpu, reason="No connection to Qaptiva Access.")
def test_run_sequence_with_emulator(schedule_seq):
    """Test simulation of a Sequence using AnalogQPU."""
    schedule, seq = schedule_seq
    aqpu = IsingAQPU.from_sequence(seq, qpu=myqlm_analog_qpu)
    # Simulate a Job converted from the Sequence on AnalogQPU
    job = IsingAQPU.convert_sequence_to_job(seq)
    analog_results = aqpu.submit(job).join()
    out_analog_results = {
        sample.state.__str__(): sample.probability for sample in analog_results
    }
    assert out_analog_results == {
        "|000>": 0.999775312857816,
        "|001>": 5.1145664077504764e-05,
        "|010>": 5.114566407750468e-05,
        "|011>": 3.27402534249787e-09,
        "|100>": 5.1145664077504696e-05,
        "|101>": 3.2740253424978696e-09,
        "|110>": 3.2740253424978717e-09,
        "|111>": 7.762132597284945e-13,
    }
    # Simulate the Schedule
    job_from_sch = schedule.to_job()
    sch_results = aqpu.submit(job_from_sch(u=0)).join()
    out_sch_results = {
        sample.state.__str__(): sample.probability for sample in sch_results
    }
    # The results obtained with the schedule are close
    assert out_sch_results == {
        "|000>": 0.9997776883446927,
        "|001>": 5.296836656346037e-05,
        "|010>": 5.296836656346041e-05,
        "|011>": 3.337723293535312e-09,
        "|100>": 5.2968366563460384e-05,
        "|101>": 3.33772329353531e-09,
        "|110>": 3.337723293535311e-09,
        "|111>": 7.6860463034749e-13,
    }


class MockResponse:
    """An object similar to an output of requests.get or requests.post."""

    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    @property
    def text(self):
        return ""


def mocked_requests_get_success(*args, **kwargs):
    """Mocks a requests.get response from a working system with successful jobs."""
    operational_url = "http://fresneldevice/api/latest/system/operational"
    job_url = "http://fresneldevice/api/latest/jobs/1"
    if args[0] == operational_url if args else kwargs["url"] == operational_url:
        return MockResponse({"data": {"operational_status": "UP"}}, 200)
    elif args[0] == job_url if args else kwargs["url"] == job_url:
        return MockResponse(
            {
                "data": {
                    "status": "DONE",
                    "result": json.dumps({"counter": {"000": 0.999, "100": 0.001}}),
                }
            },
            200,
        )
    return MockResponse(None, 404)


def mocked_requests_get_non_operational(*args, **kwargs):
    """Mocks a requests.get response from a non-working system."""
    operational_url = "http://fresneldevice/api/latest/system/operational"
    job_url = "http://fresneldevice/api/latest/jobs/1"
    if args[0] == operational_url if args else kwargs["url"] == operational_url:
        return MockResponse({"data": {"operational_status": "DOWN"}}, 200)
    elif args[0] == job_url if args else kwargs["url"] == job_url:
        return MockResponse({"data": {"status": "ERROR"}}, 200)
    return MockResponse(None, 404)


def mocked_requests_get_error(*args, **kwargs):
    """Mocks a requests.get response from a working system with non-working jobs."""
    operational_url = "http://fresneldevice/api/latest/system/operational"
    job_url = "http://fresneldevice/api/latest/jobs/1"
    if args[0] == operational_url if args else kwargs["url"] == operational_url:
        return MockResponse({"data": {"operational_status": "UP"}}, 200)
    elif args[0] == job_url if args else kwargs["url"] == job_url:
        return MockResponse({"data": {"status": "ERROR"}}, 200)
    return MockResponse(None, 404)


def mocked_requests_post_success(*args, **kwargs):
    """Mocks a response to the post of a job accepted by the system."""
    job_url = "http://fresneldevice/api/latest/jobs"
    if args[0] == job_url if args else kwargs["url"] == job_url:
        if list(kwargs["json"].keys()) != ["nb_run", "pulser_sequence"]:
            return MockResponse(None, 400)
        return MockResponse({"data": {"status": "PENDING", "uid": 1}}, 200)
    return MockResponse(None, 404)


def mocked_requests_post_fail(*args, **kwargs):
    """Mocks a response to the post of a job not accepted by the system."""
    job_url = "http://fresneldevice/api/latest/jobs"
    if args[0] == job_url if args else kwargs["url"] == job_url:
        if set(kwargs["json"].keys()) != ["nb_run", "pulser_sequence"]:
            return MockResponse(None, 400)
        return MockResponse({"data": {"status": "ERROR", "uid": 1}}, 500)
    return MockResponse(None, 404)


def _switch_seq_device(seq, device):
    if device != TEMP_DEVICE:
        if device == MockDevice:
            with pytest.warns(UserWarning, match="Switching to a device"):
                seq = seq.switch_device(device)
        else:
            seq = seq.switch_device(device)
    return seq


base_uris = ["http://fresneldevice/api", None]


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_success
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post",
    side_effect=mocked_requests_post_success,
)
@pytest.mark.parametrize("base_uri", base_uris)
@pytest.mark.parametrize("remote_fresnel", [False, True])
def test_job_submission(mock_get, mock_post, base_uri, remote_fresnel, schedule_seq):
    """Test submission of Jobs to a FresnelQPU interfacing a working QPU."""
    global port
    # Can't connect with a wrong address
    with pytest.raises(QPUException, match="Connection with API failed"):
        FresnelQPU(base_uri="")

    fresnel_qpu = FresnelQPU(base_uri=base_uri)

    # Deploy the QPU on a Qaptiva server
    if remote_fresnel:
        port += 1
        server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, port))
        server_thread.daemon = True
        server_thread.start()

    # Can't submit if the Device of the Sequence does not match the TEMP_DEVICE
    _, seq = schedule_seq
    mock_seq = _switch_seq_device(seq, MockDevice)
    job_from_seq = IsingAQPU.convert_sequence_to_job(mock_seq)
    qpu = get_remote_qpu(port) if remote_fresnel else fresnel_qpu
    with pytest.raises(QPUException, match="The Sequence in job.schedule._other"):
        qpu.submit(job_from_seq)

    # Can't simulate if Register is not from calibrated Layouts
    seq = Sequence(Register.triangular_lattice(2, 2, spacing=5), TEMP_DEVICE)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(Pulse.ConstantPulse(100, 1.0, 0.0, 0.0), "rydberg_global")
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(QPUException, match="The Register of the Sequence"):
        qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_success
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post",
    side_effect=mocked_requests_post_success,
)
@pytest.mark.parametrize("base_uri", base_uris)
@pytest.mark.parametrize("remote_fresnel", [False, True])
@pytest.mark.parametrize("device", [TEMP_DEVICE, TEMP_DEVICE.to_virtual()])
def test_job_simulation(
    mock_get,
    mock_post,
    base_uri,
    device,
    remote_fresnel,
    schedule_seq,
):
    """Test Sequence simulation on a FresnelQPU interfacing a working QPU."""
    global port
    np.random.seed(123)

    # Modify the device of the Sequence
    _, seq = schedule_seq
    seq = _switch_seq_device(seq, device)
    fresnel_qpu = FresnelQPU(base_uri=base_uri)

    # Deploy the QPU on a Qaptiva server
    if remote_fresnel:
        port += 1
        server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, port))
        server_thread.daemon = True
        server_thread.start()
    qpu = get_remote_qpu(port) if remote_fresnel else fresnel_qpu

    # Simulate Sequence using Pulser Simulation
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    result = qpu.submit(job_from_seq)
    exp_result = [
        (Sample(probability=0.999, state=0), "|000>"),
        (Sample(probability=0.001, state=4), "|100>"),
    ]
    compare_results_raw_data(result.raw_data, exp_result)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get",
    side_effect=mocked_requests_get_non_operational,
)
def test_non_operational_qpu(mock_get, schedule_seq):
    """Test a FresnelQPU interfacing a non-operational QPU."""
    global port
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    assert not fresnel_qpu.is_operational
    with pytest.warns(UserWarning, match="QPU not operational,"):
        fresnel_qpu.check_system()
    # Deploy the QPU on a Qaptiva server
    port += 1
    server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, port))
    server_thread.daemon = True
    with pytest.warns(UserWarning, match="QPU not operational,"):
        server_thread.start()
    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(QPUException, match="QPU not operational,"):
        fresnel_qpu.submit(job_from_seq)
    remote_qpu = get_remote_qpu(port)
    with pytest.raises(QPUException, match="QPU not operational,"):
        remote_qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_success
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post", side_effect=mocked_requests_post_fail
)
@pytest.mark.parametrize("remote_fresnel", [False, True])
def test_submission_error(mock_get, mock_post, remote_fresnel, schedule_seq):
    """Test a FresnelQPU interfacing a working QPU that fails at launching jobs."""
    global port
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    if remote_fresnel:
        port += 1
        server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, port))
        server_thread.daemon = True
        server_thread.start()
    qpu = get_remote_qpu(port) if remote_fresnel else fresnel_qpu
    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(QPUException, match="Could not create job"):
        qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_error
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post",
    side_effect=mocked_requests_post_success,
)
@pytest.mark.parametrize("remote_fresnel", [False, True])
def test_execution_error(mock_get, mock_post, remote_fresnel, schedule_seq):
    """Test a FresnelQPU interfacing a non-working QPU which could accept jobs."""
    global port
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    if remote_fresnel:
        port += 1
        server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, port))
        server_thread.daemon = True
        server_thread.start()
    qpu = get_remote_qpu(port) if remote_fresnel else fresnel_qpu
    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(QPUException, match="An error occured,"):
        qpu.submit(job_from_seq)

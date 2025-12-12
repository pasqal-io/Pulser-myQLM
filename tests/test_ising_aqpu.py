from __future__ import annotations

import json
from collections import Counter
from importlib.metadata import version

import numpy as np
import pytest
from pulser import Pulse, Sequence
from pulser.channels import Raman, Rydberg
from pulser.devices import MockDevice, VirtualDevice
from pulser.devices.interaction_coefficients import c6_dict
from pulser.waveforms import CustomWaveform
from pulser_simulation import QutipEmulator
from qat.comm.exceptions.ttypes import QPUException
from qat.core import Batch, Job, Result, Sample, Schedule
from qat.core.variables import ArithExpression, Symbol, cos, sin
from qat.qpus import PyLinalg

from pulser_myqlm.ising_aqpu import IsingAQPU
from pulser_myqlm.myqlmtools import are_equivalent_schedules, sample_schedule
from tests.helpers.compare_raw_data import compare_results_raw_data


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
            # c6 int btw qbit i and j adds -c6_interaction / 4 to coeff in front of Z_i
            assert (
                term._coeff.get_value()
                == -np.sum(test_ising_qpu.c6_interactions[term.qbits[0]][:]) / 4.0
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
        phase = amp / 2.0
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
        y_coeff = 0.5 * sin(phase) * amp
        assert set(dict_terms["X"].values()) == {
            x_coeff if not isinstance(x_coeff, ArithExpression) else x_coeff.to_thrift()
        }
        assert set(dict_terms["Y"].values()) == {
            y_coeff if not isinstance(y_coeff, ArithExpression) else y_coeff.to_thrift()
        }

    # Z coefficients are associated to det
    if det == 0:
        assert len(dict_terms["Z"]) == 0
        assert obs.constant_coeff == 0.0
    else:
        assert len(dict_terms["Z"]) == test_ising_qpu.nbqubits
        assert len(set(dict_terms["Z"].values())) == 1
        z_coeff = det / 2.0
        assert set(dict_terms["Z"].values()) == {
            z_coeff if not isinstance(z_coeff, ArithExpression) else z_coeff.to_thrift()
        }
        assert obs.constant_coeff == -det / 2.0


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
            z_coeff = dict_ising_int["Z"][qbits] + det / 2.0
            assert (
                term_coeff == z_coeff.to_thrift()
                if isinstance(z_coeff, ArithExpression)
                else z_coeff
            )


@pytest.mark.parametrize("device_type", ["raman", "local"])
def test_convert_init_sequence_to_schedule(test_ising_qpu, device_type):
    """Testing IsingAQPU.convert_sequence_to_schedule."""
    # An empty sequence returns a Schedule of duration 0
    seq = Sequence(test_ising_qpu.register, MockDevice)
    assert 0 == IsingAQPU.convert_sequence_to_schedule(seq).tmax
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


@pytest.mark.parametrize("aggregate_data", [True, False])
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
    ],
)
def test_conversion_sampling_result(
    meta_data, err_mess, schedule_seq, test_ising_qpu, aggregate_data
):
    """Test the conversion of MyQLM Result into a Counter as in Pulser."""
    np.random.seed(111)
    _, seq = schedule_seq
    sim_result = QutipEmulator.from_sequence(seq).run()
    n_samples = 1000
    sim_samples = sim_result.sample_final_state(n_samples)
    sim_samples_dict = {k: v for k, v in sim_samples.items()}
    # Testing the conversion of the pulser samples in a myqlm result
    myqlm_result = test_ising_qpu.convert_samples_to_result(sim_samples, aggregate_data)
    myqlm_result_from_dict = test_ising_qpu.convert_samples_to_result(
        sim_samples_dict, aggregate_data
    )
    assert myqlm_result.meta_data["n_samples"] == str(
        n_samples
    ) and myqlm_result.meta_data["n_qubits"] == str(test_ising_qpu.nbqubits)
    if not aggregate_data:
        assert len(myqlm_result.raw_data) == n_samples
    myqlm_samples = {}
    for sample in myqlm_result.raw_data:
        if sample.state.int not in myqlm_samples:
            myqlm_samples[sample.state.int] = 0
        myqlm_samples[sample.state.int] += sample.probability
    # acount for float imprecision
    for sample, proba in myqlm_samples.items():
        myqlm_samples[sample] = round(proba, 6)
    myqlm_samples_from_dict = {}
    for sample in myqlm_result_from_dict.raw_data:
        if sample.state.int not in myqlm_samples_from_dict:
            myqlm_samples_from_dict[sample.state.int] = 0
        myqlm_samples_from_dict[sample.state.int] += sample.probability
    # acount for float imprecision
    for sample, proba in myqlm_samples_from_dict.items():
        myqlm_samples_from_dict[sample] = round(proba, 6)
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
    myqlm_version = tuple(map(int, version("myqlm").split(".")))
    assert (
        (schedule_from_seq.to_job().nbshots == 0)
        if myqlm_version > (1, 9, 9)
        else (schedule_from_seq.to_job().nbshots is None)
    )
    assert job_from_seq.schedule._other == schedule_from_seq._other
    # converting a parametrized sequence fails
    amp = seq.declare_variable("amp", dtype=float)
    seq.add(Pulse.ConstantPulse(1000, amp, 0, 0), "ryd_glob")
    with pytest.raises(
        NotImplementedError, match="Parametrized sequences can't be sampled."
    ):
        IsingAQPU.convert_sequence_to_job(seq, modulation=modulation)
    # Can't simulate a sequence with a mappable register
    mappable_seq = Sequence(
        seq.register.layout.make_mappable_register(n_qubits=10), seq.device
    )
    mappable_seq.declare_channel("rydberg_global", "rydberg_global")
    mappable_seq.add(Pulse.ConstantPulse(1000, np.pi, 0, 0), "rydberg_global")
    with pytest.raises(
        RuntimeError,
        match="Can't access the sequence's register because the register is mappable",
    ):
        IsingAQPU.convert_sequence_to_job(mappable_seq)


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
def test_job_deserialization_ising(schedule_seq, other_value, err_mess):
    """Test value of Job.schedule._other for the result of a Sequence conversion."""
    schedule, seq = schedule_seq
    aqpu = IsingAQPU.from_sequence(seq)
    job = schedule.to_job()
    job.schedule._other = other_value
    with pytest.raises(ValueError, match=err_mess):
        aqpu.submit(job)


@pytest.mark.parametrize("aggregate_data", (True, False))
def test_run_sequence_ising(schedule_seq, circuit_job, aggregate_data):
    """Test simulation of a Sequence using pulser-simulation."""
    np.random.seed(111)
    schedule, seq = schedule_seq
    aqpu = IsingAQPU.from_sequence(seq)
    # IsingQPU can only run job with a schedule
    # Defining a job from a circuit instead of a schedule
    with pytest.raises(
        QPUException, match="IsingAQPU can only execute a schedule job."
    ):
        aqpu.submit(circuit_job)

    # Run job created from a sequence using convert_sequence_to_job
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq, nbshots=1000)
    job_from_seq.aggregate_data = aggregate_data
    assert job_from_seq.nbshots == 1000
    result = aqpu.submit(job_from_seq)
    exp_result = (
        [(Sample(probability=0.999, state=0), "|000>")]
        if aggregate_data
        else [(Sample(probability=0.001, state=0), "|000>") for _ in range(999)]
    ) + [(Sample(probability=0.001, state=1), "|001>")]
    compare_results_raw_data(result.raw_data, exp_result)
    assert IsingAQPU.convert_result_to_samples(result) == {"000": 999, "001": 1}
    # Run Batch of Jobs
    np.random.seed(111)
    result = aqpu.submit(Batch([job_from_seq]), meta_data={"name": "Dummy"})
    result.meta_data == {"name": "Dummy"}
    compare_results_raw_data(result[0].raw_data, exp_result)
    assert IsingAQPU.convert_result_to_samples(result[0]) == {"000": 999, "001": 1}
    # Run job created from a sequence using convert_sequence_to_schedule
    schedule_from_seq = aqpu.convert_sequence_to_schedule(seq)
    job_from_seq = schedule_from_seq.to_job()  # manually defining number of shots
    job_from_seq.aggregate_data = aggregate_data
    assert not job_from_seq.nbshots
    result_schedule = aqpu.submit(job_from_seq)
    exp_result_schedule = (
        [(Sample(probability=1.0, state=0), "|000>")]
        if aggregate_data
        else [(Sample(probability=0.0005, state=0), "|000>") for _ in range(2000)]
    )
    compare_results_raw_data(result_schedule.raw_data, exp_result_schedule)
    assert IsingAQPU.convert_result_to_samples(result_schedule) == {"000": 2000}

    # Can simulate Job if Schedule is not equivalent to Sequence
    empty_job = Job()
    empty_schedule = Schedule()
    empty_schedule._other = schedule_from_seq._other
    empty_job.schedule = empty_schedule
    empty_job.aggregate_data = aggregate_data
    result_empty_sch = aqpu.submit(empty_job)
    exp_result_empty_sch = (
        [(Sample(probability=0.9995, state=0), "|000>")]
        if aggregate_data
        else [(Sample(probability=0.0005, state=0), "|000>") for _ in range(1999)]
    )
    exp_result_empty_sch += [(Sample(probability=0.0005, state=2), "|010>")]
    compare_results_raw_data(result_empty_sch.raw_data, exp_result_empty_sch)
    assert IsingAQPU.convert_result_to_samples(result_empty_sch) == {
        "000": 1999,
        "010": 1,
    }

    # Submit_job of IsingAQPU must not be used if qpu is not None
    aqpu.set_qpu(PyLinalg())
    with pytest.raises(
        ValueError,
        match="`submit_job` must not be used if the qpu attribute is defined,",
    ):
        aqpu.submit_job(schedule.to_job())

    # Can't simulate a time-dependent job with PyLinalg (a simulator of circuits)
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
    res_dict = {
        "|000>": 0.9993103713371461,
        "|001>": 0.00023046712231041623,
        "|010>": 0.00023046712231041629,
        "|011>": 1.1642314174585867e-08,
        "|100>": 0.00023046712231041626,
        "|101>": 1.1642314174585877e-08,
        "|110>": 1.1642314174585875e-08,
        "|111>": 6.545006475814311e-13,
    }
    assert list(out_analog_results.keys()) == list(res_dict.keys())
    assert np.all(
        np.isclose(
            list(out_analog_results.values()),
            [res_dict[key] for key in out_analog_results],
        )
    )
    # Simulate the Schedule
    job_from_sch = schedule.to_job()
    sch_results = aqpu.submit(job_from_sch(u=0)).join()
    out_sch_results = {
        sample.state.__str__(): sample.probability for sample in sch_results
    }
    # The results obtained with the schedule are close
    res_dict = {
        "|000>": 0.9992998383147459,
        "|001>": 0.000234336602529663,
        "|010>": 0.000234336602529663,
        "|011>": 1.3149858582460545e-08,
        "|100>": 0.000234336602529663,
        "|101>": 1.314985858246052e-08,
        "|110>": 1.3149858582460543e-08,
        "|111>": 6.725982232286964e-13,
    }
    assert list(out_sch_results.keys()) == list(res_dict.keys())
    assert np.all(
        np.isclose(
            list(out_sch_results.values()), [res_dict[key] for key in out_sch_results]
        )
    )

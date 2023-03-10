import numpy as np
import pytest
from pulser import Pulse, Sequence
from pulser.channels import Raman, Rydberg
from pulser.devices import VirtualDevice
from pulser.devices.interaction_coefficients import c6_dict
from pulser.waveforms import CustomWaveform
from qat.core import Job, Schedule
from qat.core.variables import cos, sin

from pulser_myqlm import IsingAQPU
from pulser_myqlm.myqlmtools import PSchedule, are_equivalent_schedules, mod


def test_nbqubits(test_pulser_qpu):
    assert test_pulser_qpu.nbqubits == 4


def test_distances(test_pulser_qpu):
    dist_tl = np.array(
        [
            [0, 0, 0, 0],
            [4, 0, 0, 0],
            [4, 4 * np.sqrt(2), 0, 0],
            [4 * np.sqrt(2), 4, 4, 0],
        ]
    )
    assert np.all(test_pulser_qpu.distances == dist_tl + dist_tl.T)


def test_submit_job_pulser_qpu(test_pulser_qpu):
    with pytest.raises(
        NotImplementedError,
        match="Submit job only implemented for hardware-specific qpus, not PulserAQPU.",
    ):
        test_pulser_qpu.submit_job(Job())


def test_ising_init(test_ising_qpu):
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
    int_edge = c6_dict[test_ising_qpu.device.rydberg_level] / 4**6
    int_diag = c6_dict[test_ising_qpu.device.rydberg_level] / (4 * np.sqrt(2)) ** 6
    int_tl = np.array(
        [
            [0, 0, 0, 0],
            [int_edge, 0, 0, 0],
            [int_edge, int_diag, 0, 0],
            [int_diag, int_edge, int_edge, 0],
        ]
    )
    assert np.all(test_ising_qpu.c6_interactions == int_tl + int_tl.T)


def test_interaction_observables(test_ising_qpu):
    assert test_ising_qpu.nbqubits == test_ising_qpu.interaction_observables.nbqbits
    # Testing the coefficient in front of each operator
    # Each c6 interaction adds c6_interaction / 4 to coeff in front of I
    assert np.isclose(
        test_ising_qpu.interaction_observables._constant_coeff.get_value(),
        np.sum(np.tril(test_ising_qpu.c6_interactions) / 4.0),
        rtol=1e-15,
    )
    for term in test_ising_qpu.interaction_observables._terms:
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
    def phase_test(dict_terms, index):
        if index == "X":
            opp_index = "Y"
        else:
            opp_index = "X"
        assert len(dict_terms[opp_index]) == 0
        assert len(dict_terms[index]) == test_ising_qpu.nbqubits
        assert len(set(dict_terms[index].values())) == 1
        assert set(dict_terms[index].values()) == {(-1) ** (index == "Y") * amp / 2.0}

    amp, det, phase = (
        request.getfixturevalue(pulse_attr)
        if isinstance(pulse_attr, str)
        else pulse_attr
        for pulse_attr in (amp, det, phase)
    )

    obs = test_ising_qpu.pulse_observables(amp, det, phase)
    assert test_ising_qpu.nbqubits == obs.nbqbits
    assert obs._constant_coeff.get_value() == 0.0
    dict_terms = {"X": {}, "Y": {}, "Z": {}}
    for term in obs._terms:
        assert term.op in dict_terms.keys()
        dict_terms[term.op][term.qbits[0]] = term._coeff.get_value()

    if amp == 0:
        assert len(dict_terms["X"]) == 0
        assert len(dict_terms["Y"]) == 0
    elif mod(phase, np.pi) == 0:
        # 0 Y observables, nqubits X observables
        phase_test(dict_terms, "X")
    elif mod(phase, np.pi) == np.pi / 2:
        # 0 X observables, nqubits Y observables
        phase_test(dict_terms, "Y")
    else:
        assert (
            len(dict_terms["X"]) == len(dict_terms["Y"])
            and len(dict_terms["X"]) == test_ising_qpu.nbqubits
        )
        assert (
            len(set(dict_terms["X"].values())) == 1
            and len(set(dict_terms["Y"].values())) == 1
        )
        assert set(dict_terms["X"].values()) == {cos(phase) * amp * 0.5}
        assert set(dict_terms["Y"].values()) == {-sin(phase) * 0.5 * amp}

    if det == 0:
        assert len(dict_terms["Z"]) == 0
    else:
        assert len(dict_terms["Z"]) == test_ising_qpu.nbqubits
        assert len(set(dict_terms["Z"].values())) == 1
        assert set(dict_terms["Z"].values()) == {-(det / 2.0)}


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
    amp, det, phase = (
        request.getfixturevalue(pulse_attr)
        if isinstance(pulse_attr, str)
        else pulse_attr
        for pulse_attr in (amp, det, phase)
    )
    ising_ham = test_ising_qpu.hamiltonian(amp, det, phase)
    if (amp, det, phase) == (0, 0, 0):
        assert ising_ham == test_ising_qpu.interaction_observables
    else:
        dict_terms = {"X": {}, "Y": {}, "Z": {}, "ZZ": {}}
        for term in ising_ham.terms:
            assert term.op in dict_terms.keys()
            dict_terms[term.op][tuple(term.qbits)] = term._coeff.get_value()

        dict_ising_int = {"Z": {}, "ZZ": {}}
        for term in test_ising_qpu.interaction_observables.terms:
            dict_ising_int[term.op][tuple(term.qbits)] = term._coeff.get_value()

        for qbits, term_coeff in dict_terms["Z"].items():
            assert qbits in dict_ising_int["Z"].keys()
            assert term_coeff == dict_ising_int["Z"][qbits] + -(det / 2.0)


@pytest.mark.parametrize("device_type", ["raman", "local"])
def test_convert_init_sequence_to_schedule(test_ising_qpu, device_type):
    # Which is equivalent to having defined pulses using a Sequence
    seq = Sequence(test_ising_qpu.register, test_ising_qpu.device)
    assert PSchedule() == IsingAQPU.convert_sequence_to_schedule(seq)
    assert Schedule() == IsingAQPU.convert_sequence_to_schedule(seq, asPSchedule=False)
    if device_type == "raman":
        seq.declare_channel("ram_glob", "raman_global")
        with pytest.raises(
            TypeError,
            match="Declared channel is not Rydberg.",
        ):
            IsingAQPU.convert_sequence_to_schedule(seq)
    elif device_type == "local":
        seq.declare_channel("ryd_loc", "rydberg_local")
        with pytest.raises(
            TypeError,
            match="Declared channel is not Rydberg.Global.",
        ):
            IsingAQPU.convert_sequence_to_schedule(seq)
    seq.declare_channel("ryd_glob", "rydberg_global")
    with pytest.raises(
        ValueError,
        match="More than one channel declared.",
    ):
        IsingAQPU.convert_sequence_to_schedule(seq)


def test_convert_sequence_to_schedule(test_ising_qpu, omega_t, delta_t):
    t0 = 16  # in ns
    H0 = test_ising_qpu.hamiltonian(omega_t, delta_t, 0)
    t1 = 20  # in ns
    H1 = test_ising_qpu.hamiltonian(1, 0, 0)
    t2 = 20  # in ns
    H2 = test_ising_qpu.hamiltonian(1, 0, np.pi / 2)

    schedule0 = Schedule(drive=[(1, H0)], tmax=t0)
    schedule1 = Schedule(drive=[(1, H1)], tmax=t1)
    schedule2 = Schedule(drive=[(1, H2)], tmax=t2)
    schedule = schedule0 | schedule1 | schedule2

    Pschedule0 = PSchedule(drive=[(1, H0)], tmax=t0)
    Pschedule1 = PSchedule(drive=[(1, H1)], tmax=t1)
    Pschedule2 = PSchedule(drive=[(1, H2)], tmax=t2)
    Pschedule = Pschedule0 | Pschedule1 | Pschedule2

    # Which is equivalent to having defined pulses using a Sequence
    seq = Sequence(test_ising_qpu.register, test_ising_qpu.device)
    seq.declare_channel("ryd_glob", "rydberg_global")

    seq.add(
        Pulse(
            CustomWaveform([omega_t(t=ti) for ti in range(t0)]),
            CustomWaveform(
                [delta_t(t=ti, u=0) for ti in range(t0)]
            ),  # no parametrized sequence for the moment
            0,
        ),
        "ryd_glob",
    )
    seq.add(Pulse.ConstantPulse(t1, 1, 0, 0), "ryd_glob")
    seq.add(Pulse.ConstantPulse(t2, 1, 0, np.pi / 2), "ryd_glob")
    Pschedule_from_seq = IsingAQPU.convert_sequence_to_schedule(seq)
    Pschedule_from_seq_dec_qpu = test_ising_qpu.convert_sequence_to_schedule(seq)
    schedule_from_seq = IsingAQPU.convert_sequence_to_schedule(seq, asPSchedule=False)
    assert isinstance(Pschedule_from_seq, PSchedule)
    assert are_equivalent_schedules(Pschedule(u=0), Pschedule_from_seq)
    assert isinstance(schedule_from_seq, Schedule)
    assert are_equivalent_schedules(Pschedule(u=0), schedule_from_seq)
    assert isinstance(Pschedule_from_seq_dec_qpu, PSchedule)
    assert are_equivalent_schedules(Pschedule(u=0), Pschedule_from_seq_dec_qpu)
    assert not are_equivalent_schedules(schedule(u=0), Pschedule_from_seq)


@pytest.mark.parametrize("modulation, extended_duration", [(False, 18), (True, 0)])
def test_convert_sequence_to_job(test_ising_qpu, modulation, extended_duration):
    # Which is equivalent to having defined pulses using a Sequence
    if modulation:
        device = VirtualDevice(
            name="VirtDevice",
            dimensions=2,
            channel_objects=(
                Rydberg("Global", max_abs_detuning=None, max_amp=None, mod_bandwidth=4),
            ),
            rydberg_level=60,
        )
        seq = Sequence(test_ising_qpu.register, device)
    else:
        seq = Sequence(test_ising_qpu.register, test_ising_qpu.device)
    seq.declare_channel("ryd_glob", "rydberg_global")
    seq.add(Pulse.ConstantPulse(16, 1, 0, 0), "ryd_glob")
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq, modulation, extended_duration)
    schedule_from_seq = IsingAQPU.convert_sequence_to_schedule(
        seq, modulation, extended_duration
    )
    assert are_equivalent_schedules(schedule_from_seq, job_from_seq.schedule)

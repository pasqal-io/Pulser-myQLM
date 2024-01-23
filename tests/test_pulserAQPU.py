import json
from collections import Counter
from threading import Thread
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
from qat.core import Result, Sample, Schedule
from qat.core.qpu import QPUHandler
from qat.core.variables import ArithExpression, Symbol, cos, sin
from qat.qpus import PyLinalg

from pulser_myqlm import FresnelDevice, FresnelQPU, IsingAQPU
from pulser_myqlm.myqlmtools import are_equivalent_schedules


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
    def phase_test(dict_terms, index):
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
        request.getfixturevalue(pulse_attr)
        if isinstance(pulse_attr, str)
        else pulse_attr
        for pulse_attr in (amp, det, phase)
    )

    obs = test_ising_qpu.pulse_observables(amp, det, phase)
    assert test_ising_qpu.nbqubits == obs.nbqbits
    assert obs.constant_coeff == 0.0
    dict_terms = {"X": {}, "Y": {}, "Z": {}}
    for term in obs.terms:
        assert term.op in dict_terms.keys()
        coeff = term._coeff.get_value()
        if isinstance(coeff, ArithExpression):
            dict_terms[term.op][term.qbits[0]] = coeff.to_thrift()
            continue
        dict_terms[term.op][term.qbits[0]] = coeff

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
        x_coeff = 0.5 * cos(phase) * amp
        y_coeff = -0.5 * sin(phase) * amp
        assert set(dict_terms["X"].values()) == {
            x_coeff if not isinstance(x_coeff, ArithExpression) else x_coeff.to_thrift()
        }
        assert set(dict_terms["Y"].values()) == {
            y_coeff if not isinstance(y_coeff, ArithExpression) else y_coeff.to_thrift()
        }

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
def schedule_seq(test_ising_qpu, omega_t, delta_t):
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
    seq.add(Pulse.ConstantPulse(t1, 1, 0, 0), "ryd_glob", protocol="no-delay")
    seq.add(Pulse.ConstantPulse(t2, 1, 0, np.pi / 2), "ryd_glob", protocol="no-delay")
    return (schedule, seq)


def test_convert_sequence_to_schedule(schedule_seq):
    schedule, seq = schedule_seq
    schedule_from_seq = IsingAQPU.convert_sequence_to_schedule(seq)
    assert isinstance(schedule_from_seq, Schedule)
    assert are_equivalent_schedules(schedule(u=0), schedule_from_seq)


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
            "State 4 is incompatible with number of qubits declared 1",
        ),
        (
            {"n_qubits": 4, "n_samples": 999},
            "Probability associated with state 0 does not",
        ),
    ],
)
def test_conversion_sampling_result(meta_data, err_mess, schedule_seq, test_ising_qpu):
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
    with pytest.raises(ValueError, match=err_mess):
        test_ising_qpu.convert_result_to_samples(myqlm_result)


@pytest.mark.parametrize("modulation", [False, True])
def test_convert_sequence_to_job(schedule_seq, modulation):
    # Which is equivalent to having defined pulses using a Sequence
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq, modulation=modulation)
    schedule_from_seq = IsingAQPU.convert_sequence_to_schedule(seq, modulation)
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
@pytest.mark.parametrize("qpu", [None, FresnelQPU(None)])
def test_job_deserialization(schedule_seq, other_value, err_mess, qpu):
    schedule, seq = schedule_seq
    aqpu = IsingAQPU.from_sequence(seq, qpu=qpu)
    job = schedule.to_job()
    job.schedule._other = other_value
    with pytest.raises(ValueError, match=err_mess):
        aqpu.submit(job)


def deploy_qpu(qpu: QPUHandler, port: int) -> None:
    qpu.serve(port, "localhost")


def compare_results_raw_data(results1: list, results2: list) -> bool:
    """Compares list of samples as in Result.raw_data."""
    for i, sample1 in enumerate(results1):
        if (sample1.probability, sample1._state) != (
            results2[i].probability,
            results2[i]._state,
        ):
            return False
    return True


@pytest.mark.parametrize("qpu", [None, "fresnel"])
def test_run_sequence(schedule_seq, qpu):
    np.random.seed(123)
    schedule, seq = schedule_seq
    sim_qpu = None
    if qpu is not None:
        sim_qpu = FresnelQPU(None)
        assert sim_qpu.is_operational
        sim_qpu.check_system()
        server_thread = Thread(target=deploy_qpu, args=(sim_qpu, 1234))
        server_thread.daemon = True
        server_thread.start()

    aqpu = IsingAQPU.from_sequence(seq, qpu=sim_qpu)
    # Run job created from a sequence
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq, nbshots=1000)
    assert job_from_seq.nbshots == 1000
    result = aqpu.submit(job_from_seq)
    exp_result = [
        Sample(probability=0.999, state=0),
        Sample(probability=0.001, state=4),
    ]
    assert compare_results_raw_data(result.raw_data, exp_result)
    # Use IsingAQPU converter to have information about the sequence in the job
    schedule_from_seq = aqpu.convert_sequence_to_schedule(seq)
    job_from_seq = schedule_from_seq.to_job()  # max nb shots
    assert not job_from_seq.nbshots
    result_schedule = aqpu.submit(job_from_seq)
    exp_result_schedule = [
        Sample(probability=0.9995, state=0),
        Sample(probability=0.0005, state=1),
    ]
    assert compare_results_raw_data(result_schedule.raw_data, exp_result_schedule)
    # Can't simulate Job if Schedule is not equivalent to Sequence
    job_from_seq.schedule = 2 * job_from_seq.schedule
    with pytest.raises(
        ValueError, match="The Sequence and the Schedule are not equivalent."
    ):
        aqpu.submit(job_from_seq)
    # Can't simulate a time-dependent job with a PyLinalg AQPU
    aqpu.set_qpu(PyLinalg())
    with pytest.raises(
        ValueError,
        match="`submit_job` must not be used if the qpu attribute is defined,",
    ):
        aqpu.submit_job(schedule.to_job())
    with pytest.raises(TypeError, match="'NoneType' object is not"):
        aqpu.submit(schedule.to_job())


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def text(self):
        return ""


def mocked_requests_get_success(*args, **kwargs):
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
    operational_url = "http://fresneldevice/api/latest/system/operational"
    job_url = "http://fresneldevice/api/latest/jobs/1"
    if args[0] == operational_url if args else kwargs["url"] == operational_url:
        return MockResponse({"data": {"operational_status": "DOWN"}}, 200)
    elif args[0] == job_url if args else kwargs["url"] == job_url:
        return MockResponse(
            {
                "data": {
                    "status": "DONE",
                    "result": json.dumps({"counter": {"000": 0.9995, "100": 0.0005}}),
                }
            },
            200,
        )
    return MockResponse(None, 404)


def mocked_requests_get_error(*args, **kwargs):
    operational_url = "http://fresneldevice/api/latest/system/operational"
    job_url = "http://fresneldevice/api/latest/jobs/1"
    if args[0] == operational_url if args else kwargs["url"] == operational_url:
        return MockResponse({"data": {"operational_status": "UP"}}, 200)
    elif args[0] == job_url if args else kwargs["url"] == job_url:
        return MockResponse({"data": {"status": "ERROR"}}, 200)
    return MockResponse(None, 404)


def mocked_requests_post_success(*args, **kwargs):
    job_url = "http://fresneldevice/api/latest/jobs"
    if args[0] == job_url if args else kwargs["url"] == job_url:
        if list(kwargs["json"].keys()) != ["nb_run", "pulser_sequence"]:
            return MockResponse(None, 400)
        return MockResponse({"data": {"status": "PENDING", "uid": 1}}, 200)
    return MockResponse(None, 404)


def mocked_requests_post_fail(*args, **kwargs):
    job_url = "http://fresneldevice/api/latest/jobs"
    if args[0] == job_url if args else kwargs["url"] == job_url:
        if set(kwargs["json"].keys()) != ["nb_run", "pulser_sequence"]:
            return MockResponse(None, 400)
        return MockResponse({"data": {"status": "ERROR", "uid": 1}}, 500)
    return MockResponse(None, 404)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_success
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post",
    side_effect=mocked_requests_post_success,
)
@pytest.mark.parametrize("base_uri", ["http://fresneldevice/api", None])
@pytest.mark.parametrize(
    "device", [FresnelDevice, FresnelDevice.to_virtual(), MockDevice]
)
def test_successful_QPU(mock_get, mock_post, base_uri, device, schedule_seq):
    np.random.seed(123)
    with pytest.raises(ValueError, match="Connection with API failed"):
        FresnelQPU(base_uri="")

    fresnel_qpu = FresnelQPU(base_uri=base_uri)

    # Deploy the QPU on a Qaptiva server
    server_thread = Thread(target=deploy_qpu, args=(fresnel_qpu, 1235))
    server_thread.daemon = True
    server_thread.start()

    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    if device != FresnelDevice:
        if device == MockDevice:
            with pytest.warns(UserWarning, match="Switching to a device"):
                seq = seq.switch_device(device)
        else:
            seq = seq.switch_device(device)

    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    if device == MockDevice:
        with pytest.raises(ValueError, match="The Sequence in job.schedule._other"):
            fresnel_qpu.submit(job_from_seq)
    else:
        result = fresnel_qpu.submit(job_from_seq)
        exp_result = [
            Sample(probability=0.999, state=0),
            Sample(probability=0.001, state=4),
        ]
        assert compare_results_raw_data(result.raw_data, exp_result)
    # Can't simulate if register is not from calibrated layout
    seq = Sequence(Register.triangular_lattice(2, 2, spacing=5), FresnelDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(Pulse.ConstantPulse(100, 1.0, 0.0, 0.0), "rydberg_global")
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(ValueError, match="The Register of the Sequence"):
        fresnel_qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get",
    side_effect=mocked_requests_get_non_operational,
)
def test_non_operational_QPU(mock_get, schedule_seq):
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    assert not fresnel_qpu.is_operational
    with pytest.raises(OSError, match="QPU not operational,"):
        fresnel_qpu.check_system()
    # Deploy the QPU on a Qaptiva server
    with pytest.raises(OSError, match="QPU not operational,"):
        fresnel_qpu.serve(1234)

    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(OSError, match="QPU not operational,"):
        fresnel_qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_success
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post", side_effect=mocked_requests_post_fail
)
def test_submission_error(mock_get, mock_post, schedule_seq):
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(Exception, match="Could not create job"):
        fresnel_qpu.submit(job_from_seq)


@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.get", side_effect=mocked_requests_get_error
)
@mock.patch(
    "pulser_myqlm.pulserAQPU.requests.post",
    side_effect=mocked_requests_post_success,
)
def test_execution_error(mock_get, mock_post, schedule_seq):
    base_uri = "http://fresneldevice/api"
    fresnel_qpu = FresnelQPU(base_uri=base_uri)
    # Simulate Sequence using Pulser Simulation
    _, seq = schedule_seq
    job_from_seq = IsingAQPU.convert_sequence_to_job(seq)
    with pytest.raises(Exception, match="An error occured,"):
        fresnel_qpu.submit(job_from_seq)

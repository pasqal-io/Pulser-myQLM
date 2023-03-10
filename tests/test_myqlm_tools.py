import numpy as np
import pytest
from qat.core import Observable, Schedule, Term
from qat.core.variables import ArithExpression, heaviside

from pulser_myqlm import Pheaviside
from pulser_myqlm.myqlmtools import (
    Pheaviside_eval,
    Pmod,
    PSchedule,
    are_equivalent_schedules,
    mod,
    sample_schedule,
)


@pytest.mark.parametrize(
    "t, heaviside_value", [(-1, 0), (0, 1), (1, 1), (5, 0), (10, 0)]
)
def test_Pheaviside(t, heaviside_value, u_variable):
    # evaluator and Symbol have same value.
    assert Pheaviside_eval(t, 0, 5) == heaviside_value
    assert Pheaviside(t, 0, 5) == heaviside_value
    # Pheaviside and heaviside don't have same value for t=upper bound.
    if t != 5:
        assert Pheaviside(t, 0, 5) == heaviside(t, 0, 5)
    else:
        assert Pheaviside(t, 0, 5) != heaviside(t, 0, 5)
    # Symbol can be defined with a Variable.
    heaviside_expr = Pheaviside(u_variable, 0, 5)
    assert isinstance(heaviside_expr, ArithExpression)
    assert heaviside_expr.get_variables() == ["u"]
    assert heaviside_expr.to_thrift() == "Pheaviside u 0 5"
    assert heaviside_expr(u=t) == heaviside_value


schedule1 = Schedule(
    drive=[
        (1.0, Observable(1, pauli_terms=[Term(1.0, "Z", [0])])),
        (1.0, Observable(1, pauli_terms=[Term(1.0, "X", [0])])),
    ],
    tmax=10,
    tname="t",
    gamma_t=None,
)
schedule2 = Schedule(
    drive=[(2.0, Observable(1, pauli_terms=[Term(1.0, "Y", [0])]))],
    tmax=20,
    tname="t",
    gamma_t=None,
)
Pschedule1 = PSchedule(
    drive=[
        (1.0, Observable(1, pauli_terms=[Term(1.0, "Z", [0])])),
        (1.0, Observable(1, pauli_terms=[Term(1.0, "X", [0])])),
    ],
    tmax=10,
    tname="t",
    gamma_t=None,
)
Pschedule2 = PSchedule(
    drive=[(2.0, Observable(1, pauli_terms=[Term(1.0, "Y", [0])]))],
    tmax=20,
    tname="t",
    gamma_t=None,
)


@pytest.mark.parametrize(
    "Psch_to_test, sch_to_compare, both_PSchedule, right_op",
    [
        (Pschedule1 | Pschedule2, schedule1 | schedule2, True, True),
        (Pschedule1 | schedule2, schedule1 | schedule2, False, True),
        (schedule2 | Pschedule1, schedule2 | schedule1, False, False),
        (Pschedule1 + Pschedule2, schedule1 + schedule2, True, True),
        (Pschedule1 + schedule2, schedule1 + schedule2, False, True),
        (schedule2 + Pschedule1, schedule1 + schedule2, False, True),
        (Pschedule1 << 1, schedule1 << 1, True, True),
        (Pschedule1 >> 1, schedule1 >> 1, True, True),
        (Pschedule1._shift(1), schedule1._shift(1), True, True),
        (Pschedule1._shift(-1), schedule1._shift(-1), True, True),
    ],
)
def test_PSchedule(Psch_to_test, sch_to_compare, both_PSchedule, right_op):
    assert isinstance(Psch_to_test, PSchedule)
    # Time composition/Merging/Time Shift of PSchedule uses Pheaviside.
    Pheaviside_index = np.array(
        [
            drive_coeff.get_value().to_thrift().find("Pheaviside")
            for drive_coeff in Psch_to_test.drive_coeffs
        ]
    )
    # If only Pschedules are involved then all terms have Pheaviside
    if both_PSchedule:
        assert np.all(Pheaviside_index >= 0)
        assert np.all(Pheaviside_index >= 0)
    # If operation of PSchedule and Schedule, Pheaviside appears on first terms
    # and heaviside afterwards.
    elif right_op:
        len_first_schedule = len(Pschedule1.drive)
        assert np.all(Pheaviside_index[:len_first_schedule] >= 0)
        # str.find returns -1 if no Pheaviside found.
        assert np.all(Pheaviside_index[len_first_schedule:] == -1)
    # If operation of Schedule and PSchedule, heaviside appears on first terms
    # and Pheaviside afterwards.
    else:
        len_first_schedule = len(Pschedule2.drive)
        assert np.all(Pheaviside_index[:len_first_schedule] == -1)
        assert np.all(Pheaviside_index[len_first_schedule:] >= 0)
    # Same drive observables for all.
    assert Psch_to_test.drive_obs == sch_to_compare.drive_obs


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


def test_sample_schedule(t_variable):
    # Test empty schedule.
    assert sample_schedule(Schedule()) == []
    assert sample_schedule(PSchedule()) == []
    # Test non-int tmax.
    with pytest.raises(
        TypeError,
        match="tmax should be an integer.",
    ):
        sample_schedule(
            PSchedule(
                drive=[(1, Observable(2, pauli_terms=[Term(2.0, "X", [0])]))], tmax=10.0
            )
        )
    # Test variable-independent schedule.
    tmax = 2
    schedule = PSchedule(
        drive=[(1, Observable(2, pauli_terms=[Term(2.0, "X", [0])]))], tmax=tmax
    )
    assert sample_schedule(schedule) == [
        Observable(2, pauli_terms=[Term(2.0, "X", [0])]) for _ in range(tmax)
    ]
    # Test time-independent schedule.
    sample_variable_schedule = sample_schedule(t_variable * schedule)
    assert len(sample_variable_schedule) == tmax
    # Coeff of operator "X" is 2 * t, coeff of other op is 0 (including op "I").
    assert [
        variable_schedule_ti._constant_coeff.get_value()
        for variable_schedule_ti in sample_variable_schedule
    ] == [0] * tmax
    assert [
        sample_variable_schedule[ti]._terms[0]._coeff.get_value()
        for ti in range(1, tmax)
    ] == [2.0 * ti for ti in range(1, tmax)]
    assert [sample_variable_schedule[ti]._terms[0].op for ti in range(1, tmax)] == [
        "X"
    ] * (tmax - 1)
    # Test schedule constant relative to a variable.
    sample_cst_schedule = sample_schedule(t_variable * schedule, "u")
    assert len(sample_cst_schedule) == tmax
    assert [
        variable_schedule_ti._constant_coeff.get_value()
        for variable_schedule_ti in sample_cst_schedule
    ] == [0] * tmax
    assert [
        sample_cst_schedule[ti]._terms[0]._coeff.get_value() for ti in range(1, tmax)
    ] == [ArithExpression.from_string("* 2.0 t")] * (tmax - 1)
    assert [sample_cst_schedule[ti]._terms[0].op for ti in range(1, tmax)] == ["X"] * (
        tmax - 1
    )


def test_equivalent_schedules(t_variable, delta_t):
    t0 = 16  # in ns
    H0 = t_variable * Observable(
        2, pauli_terms=[Term(1.0, "X", [0])]
    ) + delta_t * Observable(2, pauli_terms=[Term(1.0, "Z", [1])])
    t1 = 20  # in ns
    H1 = Observable(2, pauli_terms=[Term(2.0, "X", [0])])
    t2 = 20  # in ns
    H2 = Observable(2, pauli_terms=[Term(1.0, "YY", [0, 1])])

    schedule0 = PSchedule(drive=[(1, H0)], tmax=t0)
    schedule1 = PSchedule(drive=[(1, H1)], tmax=t1)
    schedule2 = PSchedule(drive=[(1, H2)], tmax=t2)
    schedule = schedule0 | schedule1 | schedule2

    schedule_sum = PSchedule(
        drive=[
            [Pheaviside(t_variable, 0, t0), H0],
            [Pheaviside(t_variable, t0, t1 + t0), H1],
            [Pheaviside(t_variable, t1 + t0, t1 + t0 + t2), H2],
        ],
        tmax=t1 + t0 + t2,
    )
    assert are_equivalent_schedules(schedule, schedule_sum)
    assert are_equivalent_schedules(schedule(u=0), schedule_sum(u=0))
    assert are_equivalent_schedules(schedule(t=0), schedule_sum(t=0), "u")
    assert not are_equivalent_schedules(schedule1, schedule2)
    assert not are_equivalent_schedules(schedule0, schedule1)

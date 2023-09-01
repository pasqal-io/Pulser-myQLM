import pytest
from qat.core import Observable, Schedule, Term
from qat.core.variables import ArithExpression, heaviside

from pulser_myqlm.myqlmtools import are_equivalent_schedules, sample_schedule

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


def test_sample_schedule(t_variable):
    # Test empty schedule.
    assert sample_schedule(Schedule()) == []
    # Test non-int tmax.
    with pytest.raises(
        TypeError,
        match="tmax should be an integer.",
    ):
        sample_schedule(
            Schedule(
                drive=[(1, Observable(2, pauli_terms=[Term(2.0, "X", [0])]))], tmax=10.0
            )
        )
    # Test variable-independent schedule.
    tmax = 2
    schedule = Schedule(
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
        variable_schedule_ti.constant_coeff
        for variable_schedule_ti in sample_variable_schedule
    ] == [0] * tmax
    assert [
        sample_variable_schedule[ti].terms[0]._coeff.get_value()
        for ti in range(1, tmax)
    ] == [2.0 * ti for ti in range(1, tmax)]
    assert [sample_variable_schedule[ti].terms[0].op for ti in range(1, tmax)] == [
        "X"
    ] * (tmax - 1)
    # Test schedule constant relative to a variable.
    sample_cst_schedule = sample_schedule(t_variable * schedule, "u")
    assert len(sample_cst_schedule) == tmax
    assert [
        variable_schedule_ti.constant_coeff
        for variable_schedule_ti in sample_cst_schedule
    ] == [0] * tmax
    assert [
        sample_cst_schedule[ti].terms[0]._coeff.get_value() for ti in range(1, tmax)
    ] == [ArithExpression.from_string("* 2.0 t")] * (tmax - 1)
    assert [sample_cst_schedule[ti].terms[0].op for ti in range(1, tmax)] == ["X"] * (
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

    schedule0 = Schedule(drive=[(1, H0)], tmax=t0)
    schedule1 = Schedule(drive=[(1, H1)], tmax=t1)
    schedule2 = Schedule(drive=[(1, H2)], tmax=t2)
    schedule = schedule0 | schedule1 | schedule2

    schedule_sum = Schedule(
        drive=[
            [heaviside(t_variable, 0, t0), H0],
            [heaviside(t_variable, t0, t1 + t0), H1],
            [heaviside(t_variable, t1 + t0, t1 + t0 + t2), H2],
        ],
        tmax=t1 + t0 + t2,
    )
    assert are_equivalent_schedules(schedule, schedule_sum)
    assert are_equivalent_schedules(schedule(u=0), schedule_sum(u=0))
    assert are_equivalent_schedules(schedule(t=0), schedule_sum(t=0), "u")
    assert not are_equivalent_schedules(schedule1, schedule2)
    assert not are_equivalent_schedules(schedule0, schedule1)

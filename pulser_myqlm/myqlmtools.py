"""Tools to interface MyQLM with Pulse."""
from __future__ import annotations

import numpy as np
from qat.core import Schedule
from qat.core.variables import ArithExpression, Symbol


def Pheaviside_eval(t: int, t1: int, t2: int) -> int:
    """Defines the evaluator of the heaviside function.

    Heaviside function returns 1 if a value is between two bounds.
    To be closer to a physical device, it returns 0 for the upper bound.

    Args:
        t: value at which to evaluate the function.
        t1: first bound.
        t2: second bound.

    Returns:
        value: 1 if t between t1 and t2, 0 otherwise.
    """
    if t >= min(t1, t2) and t < max(t1, t2):
        return 1
    return 0


Pheaviside = Symbol(token="Pheaviside", evaluator=Pheaviside_eval, arity=3)


def _replace_heaviside_Pheaviside(
    pre_op_schedule: Schedule, res_schedule: Schedule, op_on_right: bool = True
) -> PSchedule:
    """Replace heaviside used for ponderation by Pheaviside.

    In a Temporal composition/Merge of two Schedule or during a time translation
    of a Schedule, drive coefficients are multiplied by heaviside function.
    When PSchedule are involved, we want these functions to be Pheaviside.

    Args:
        pre_op_schedule: the schedule with which the operation is performed.
        res_schedule: the result of the operation using Schedule.operation.
        op_on_right: if pre_op_schedule is on the right of the operation.

    Returns:
        PSchedule: A PSchedule with Pheaviside functions in drive_coeffs.
    """
    # Schedule.operation returns a Schedule. Converts res_schedule into a PSchedule.
    Pres_schedule = PSchedule(
        drive=res_schedule.drive,
        tmax=res_schedule.tmax,
        tname=res_schedule.tname,
        gamma_t=res_schedule.gamma_t,
    )
    len_res_drive = len(Pres_schedule.drive_coeffs)
    len_pre_op_drive = len(pre_op_schedule.drive_coeffs)
    # If pre_op_schedule is a PSchedule, then all the heaviside have to be changed.
    # Otherwise, change only the ones that are not in drive coeffs of pre_op_schedule.
    nb_Pheaviside = (
        len_res_drive
        if isinstance(pre_op_schedule, PSchedule)
        else len_res_drive - len_pre_op_drive
    )
    # If pre_op_schedule is on the right of the operation, its drive coeffs are the
    # last ones in res_schedule.
    # Otherwise its drive coeffs are the first ones in res_schedule.
    index_start_replace = 0 if op_on_right else len_pre_op_drive
    for i in range(nb_Pheaviside):
        index_to_replace = (index_start_replace + i) % len_res_drive
        # Replace first heaviside in the drive coefficient of interest by Pheaviside.
        # This is the heaviside used for ponderation.
        drive_coeff_str = Pres_schedule.drive_coeffs[index_to_replace].string_p
        assert "heaviside" in drive_coeff_str
        Pres_schedule.drive_coeffs[index_to_replace].string_p = drive_coeff_str.replace(
            "heaviside", "Pheaviside", 1
        )
    return Pres_schedule


class PSchedule(Schedule):
    """A child class of Schedule using Pheaviside instead of heaviside.

    Redefines the operations of temporal composition, merging and time shifting.
    """

    def __or__(self, schedule: Schedule) -> PSchedule:
        """Temporal composition of a PSchedule with a Schedule.

        Same operation as in the case of the temporal composition of schedules,
        except Pheaviside are used for the ponderation instead of heaviside.
        """
        or_schedule = super().__or__(schedule)
        return _replace_heaviside_Pheaviside(schedule, or_schedule)

    def __ror__(self, schedule: Schedule) -> PSchedule:
        """Temporal composition of a Schedule with a PSchedule."""
        ror_schedule = schedule.__or__(self)
        return _replace_heaviside_Pheaviside(schedule, ror_schedule, False)

    def __add__(self, schedule: Schedule) -> PSchedule:
        """Merge of a PSchedule with a Schedule.

        Same operation as in the case of the merging of schedules,
        except Pheaviside are used for the ponderation instead of heaviside.
        """
        add_schedule = super().__add__(schedule)
        return _replace_heaviside_Pheaviside(schedule, add_schedule)

    def __radd__(self, schedule: Schedule) -> PSchedule:
        """Merge of a Schedule with a PSchedule."""
        return self.__add__(schedule)

    def _shift(self, value: int | float | ArithExpression) -> PSchedule:
        """Time translation of a PSchedule.

        Same operation as in the case of the time translation of a Schedule,
        except Pheaviside are used for the ponderation instead of heaviside.
        """
        shift_schedule = super()._shift(value)
        return _replace_heaviside_Pheaviside(PSchedule(), shift_schedule)

    def __lshift__(self, value: int | float | ArithExpression) -> PSchedule:
        """Time translation of a PSchedule in the past."""
        return self._shift(value)

    def __rshift__(self, value: int | float | ArithExpression) -> PSchedule:
        """Time translation of a PSchedule in the future."""
        return self._shift(value)


def Pmod(a: float, b: float) -> float:
    """Returns rest of euclidian division of a by b."""
    return a % b


mod = Symbol(token="%", evaluator=Pmod, arity=2)


def sample_schedule(schedule: Schedule, var_name: str = "") -> list:
    """Samples a schedule between 0 and its tmax.

    Args:
        schedule: the schedule to sample.
        var_name: name of the variable to evaluate between 0 and schedule.tmax.
            Defaults to schedule.tname.

    Returns:
        sampled_schedule: list of evaluations of schedule between 0 and schedule.tmax.
    """
    if not schedule.drive:
        return []
    elif not isinstance(schedule.tmax, int):
        raise TypeError("tmax should be an integer.")
    drive_coeffs_array = np.array(
        [drive_coeff.get_value() for drive_coeff in schedule.drive_coeffs]
    )
    drive_obs_array = np.array(schedule.drive_obs)
    summed_schedule = np.sum(drive_coeffs_array * drive_obs_array)
    if not var_name:
        var_name = schedule.tname
    if var_name not in schedule.get_variables():
        return [summed_schedule] * schedule.tmax
    else:
        return [summed_schedule(**{var_name: ti}) for ti in range(schedule.tmax)]


def are_equivalent_schedules(
    schedule1: Schedule, schedule2: Schedule, var_name: str = ""
) -> np.bool_:
    """Two schedules are equivalent if they are equal at each time step."""
    sample_schedule_1 = sample_schedule(schedule1, var_name)
    sample_schedule_2 = sample_schedule(schedule2, var_name)
    max_tmax1_tmax2 = max(schedule1.tmax, schedule2.tmax)
    padded_sample_schedule_1 = np.pad(
        sample_schedule_1,
        (0, max_tmax1_tmax2 - schedule1.tmax),
        "constant",
        constant_values=(0, 0),
    )
    padded_sample_schedule_2 = np.pad(
        sample_schedule_2,
        (0, max_tmax1_tmax2 - schedule2.tmax),
        "constant",
        constant_values=(0, 0),
    )
    return np.all(padded_sample_schedule_1 == padded_sample_schedule_2)

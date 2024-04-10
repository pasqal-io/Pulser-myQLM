"""Tools to interface MyQLM with Pulse."""

from __future__ import annotations

import numpy as np
from qat.core import Schedule


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
    tmax = schedule.tmax * 1000  # in ns
    if round(tmax, 6) % 1 != 0:
        raise TypeError("tmax*1000 should be an integer.")
    drive_coeffs_array = np.array(
        [drive_coeff.get_value() for drive_coeff in schedule.drive_coeffs]
    )
    drive_obs_array = np.array(schedule.drive_obs)
    summed_schedule = np.sum(drive_coeffs_array * drive_obs_array)
    if not var_name:
        var_name = schedule.tname
    # Time expressed in µs in MyQLM expressions
    return [summed_schedule(**{var_name: ti / 1000}) for ti in range(int(tmax))]


def are_equivalent_schedules(
    schedule1: Schedule, schedule2: Schedule, var_name: str = ""
) -> np.bool_:
    """Two schedules are equivalent if they are equal at each time step."""
    sample_schedule_1 = sample_schedule(schedule1, var_name)
    sample_schedule_2 = sample_schedule(schedule2, var_name)
    tmax1 = int(schedule1.tmax * 1000)  # int in ns
    tmax2 = int(schedule2.tmax * 1000)  # int in ns
    max_tmax1_tmax2 = max(tmax1, tmax2)
    padded_sample_schedule_1 = np.pad(
        sample_schedule_1,
        (0, max_tmax1_tmax2 - tmax1),
        "constant",
        constant_values=(0, 0),
    )
    padded_sample_schedule_2 = np.pad(
        sample_schedule_2,
        (0, max_tmax1_tmax2 - tmax2),
        "constant",
        constant_values=(0, 0),
    )
    return np.all(padded_sample_schedule_1 == padded_sample_schedule_2)

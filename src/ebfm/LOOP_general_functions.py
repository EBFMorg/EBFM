# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime, timedelta


def is_first_time_step(t: int) -> bool:
    """
    Check if the current time step is the first time step.

    Parameters:
        t (int): Current time step (0-based index).

    Returns:
        bool: True if the current time step is the first time step, False otherwise.
    """
    return t == 0


def is_final_time_step(t: int, time: dict) -> bool:
    """
    Check if the current time step is the final time step.

    Parameters:
        t (int): Current time step (0-based index).
        time (dict): Dictionary containing time-related variables, including 'tn' (total number of time steps).

    Returns:
        bool: True if the current time step is the final time step, False otherwise.
    """
    return t == time["tn"] - 1


def print_time(t: int, ts: datetime, dt: float) -> datetime:
    """
    Sets time parameters and prints the current model time to the screen.

    Parameters:
        t (int): Current time step.
        ts (datetime): Start time.
        dt (float): Time step size.

    Returns:
        t_cur: current time
    """
    # Calculate current time based on the time step
    t_cur = ts + timedelta(days=t * dt)

    # Extract year, month, day, and hour from the current time
    tempdate = t_cur.timetuple()

    # Print the current time parameters
    print(
        f"Year: {tempdate.tm_year:4d}   "
        f"Month: {tempdate.tm_mon:2d}   "
        f"Day: {tempdate.tm_mday:2d}   "
        f"Hour: {tempdate.tm_hour:2d}"
    )

    return t_cur

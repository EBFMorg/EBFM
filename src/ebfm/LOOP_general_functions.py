# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime, timedelta


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
    t_cur = ts + timedelta(days=(t - 1) * dt)

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

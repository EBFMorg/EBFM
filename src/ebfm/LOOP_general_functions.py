from datetime import datetime, timedelta

from netCDF4 import date2num


def print_time(t, time):
    """
    Sets time parameters and prints the current model time to the screen.

    Parameters:
        t (int): Current time step.
        time (dict): Dictionary containing time-related parameters, such as start time (ts), end time (te), and time step (dt).

    Returns:
        dict: Updated time dictionary with calculated current and previous times.
    """
    # Convert time parameters to datetime objects
    # time['TS'] = datetime.strptime(time['ts'], '%Y-%m-%d %H:%M:%S')  # Start time
    # time['TE'] = datetime.strptime(time['te'], '%Y-%m-%d %H:%M:%S')  # End time

    # Calculate current and previous model times based on the time step
    time["TCUR"] = time["ts"] + timedelta(days=(t - 1) * time["dt"])
    time["TPREV"] = time["ts"] + timedelta(days=(t - 2) * time["dt"])

    # Extract year, month, day, and hour from the current time
    tempdate = time["TCUR"].timetuple()

    # Print the current time parameters
    print(
        f"Year: {tempdate.tm_year:4d}   Month: {tempdate.tm_mon:2d}   Day: {tempdate.tm_mday:2d}   Hour: {tempdate.tm_hour:2d}"
    )

    return time

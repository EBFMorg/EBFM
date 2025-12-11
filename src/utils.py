# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from mpi4py import MPI
import sys
import logging

default_stdout_log_level = logging.INFO  # log level for logging to console
default_file_log_level = logging.DEBUG  # log level for logging to file

# if individual ranks should have different log levels to console, define them here
custom_stdout_log_levels = {
    0: logging.INFO,  # log level for rank 0
    # 1: logging.DEBUG,  # to log other ranks to console define log level here
}


def setup_logging(
    log_levels=custom_stdout_log_levels, comm=MPI.COMM_WORLD, file=None, file_log_level=default_file_log_level
):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(
        fmt=f"EBFM r{comm.rank} " + "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if file:
        file_handler = logging.FileHandler(filename=(f"{file}" + f".{comm.rank}" if comm.size > 1 else ""))
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_log_level)
        root_logger.addHandler(file_handler)

    if comm.rank in log_levels.keys():
        use_level = log_levels[comm.rank]
    else:
        use_level = default_stdout_log_level

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    stdout_handler.setLevel(use_level)
    root_logger.addHandler(stdout_handler)

    # log errors to console for all ranks
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(log_formatter)
    stderr_handler.setLevel(logging.ERROR)
    root_logger.addHandler(stderr_handler)

    root_logger.debug("Logging setup complete.")

    return logging

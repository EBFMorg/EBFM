# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from mpi4py import MPI
import sys
from pathlib import Path
import logging
from logging import Logger
from logging import getLogger  # noqa: F401

log_levels_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

default_stdout_log_level = logging.INFO  # log level for logging to console
default_file_log_level = logging.DEBUG  # log level for logging to file

# if individual ranks should have different log levels to console, define them here
default_stdout_individual_log_levels = {
    # 0: logging.INFO,  # log level for rank 0
    # 1: logging.DEBUG,  # to log other ranks to console define log level here
}


def setup_logging(
    stdout_log_level=default_stdout_log_level,
    stdout_individual_log_levels=default_stdout_individual_log_levels,
    comm=MPI.COMM_WORLD,
    file: Path = None,
    file_log_level=default_file_log_level,
):
    """
    Setup logging for EBFM with MPI support.

    @param stdout_log_level: Log level for stdout
    @param stdout_individual_log_levels: Dictionary mapping MPI ranks to their desired log levels for console output.
    @param comm: MPI communicator to use for determining rank and size.
    @param file: Optional path of log file. If None, no file logging is set up. If parallel run with multiple ranks,
                 rank number is appended to filename.
    @param file_log_level: Log level for file logging.
    """

    root_logger: Logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    is_parallel = comm.size > 1
    always_include_rank_info = False

    # include rank info in file name if parallel run or forced by configuration
    include_rank_info = is_parallel or always_include_rank_info

    if include_rank_info:
        rank_info = f"r{comm.rank} "
    else:  # keep empty for serial run
        rank_info = ""

    log_formatter = logging.Formatter(
        fmt="EBFM " + rank_info + "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if file:
        if include_rank_info:
            file = Path(f"{file}.{comm.rank}")

        file_handler = logging.FileHandler(filename=file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_log_level)
        root_logger.addHandler(file_handler)

    if comm.rank in stdout_individual_log_levels.keys():  # check for custom log level for this rank
        use_level = stdout_individual_log_levels[comm.rank]
    else:  # use general log level if no individual log level is defined for this rank
        use_level = stdout_log_level

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

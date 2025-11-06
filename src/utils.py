# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from mpi4py import MPI
import sys
import logging

default_levels = {
    "file": logging.DEBUG,  # log level for logging to file
    0: logging.INFO,  # log level for rank 0
    # 1: logging.DEBUG,  # to log other ranks to console define log level here
}


def setup_logging(log_levels=default_levels, file=None):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(
        fmt=f"EBFM r{MPI.COMM_WORLD.rank} "
        + "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if file:
        file_handler = logging.FileHandler(
            filename=(f"{file}" + f".{MPI.COMM_WORLD.rank}" if MPI.COMM_WORLD.size > 1 else "")
        )
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(log_levels["file"])
        root_logger.addHandler(file_handler)

    if MPI.COMM_WORLD.rank in log_levels.keys():
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_formatter)
        stdout_handler.setLevel(log_levels[MPI.COMM_WORLD.rank])
        root_logger.addHandler(stdout_handler)

    # log errors to console for all ranks
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(log_formatter)
    stderr_handler.setLevel(logging.ERROR)
    root_logger.addHandler(stderr_handler)

    root_logger.info("Logging setup complete.")

    return logging

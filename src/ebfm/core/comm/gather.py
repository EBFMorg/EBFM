# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect the columns that the ranks of EBFM's communicator hold on one rank, e.g. to write them to a file, and
distribute them again, e.g. after reading them from a file."""

import numpy as np
from numpy.typing import NDArray

from ebfm.core.comm import Comm


class ColumnGather:
    """
    Gathers per-column arrays of all ranks on a root rank, ordered by column id, and scatters them back.

    Every rank holds a subset of the columns, identified by their ids (grid["column_id"], see ebfm.core.grid).
    The root rank receives the columns of all ranks sorted by id.

    Construction and every gather and scatter are collective: all ranks of the communicator must take part, in the
    same order. With a single rank, nothing is communicated and they return their input unchanged.
    """

    def __init__(self, comm: Comm, column_ids: NDArray[np.integer], root: int = 0):
        """
        @param[in] comm communicator whose ranks hold the columns
        @param[in] column_ids ids of the columns this rank holds (grid["column_id"]); no other rank may hold them
                              (see ebfm.core.grid.validate_grid_distribution)
        @param[in] root rank that receives the gathered arrays
        """
        self.comm = comm
        self.root = root
        self.is_root = comm.rank == root
        self.n_local_columns = len(column_ids)

        if comm.size == 1:
            self.column_ids = column_ids
            self._order = None
            return

        self._counts = comm.gather(self.n_local_columns, root=root)
        all_column_ids = self.gather_unordered(column_ids)
        if self.is_root:
            self._order = np.argsort(all_column_ids, kind="stable")
            self.column_ids = all_column_ids[self._order]
        else:
            self._order = None
            self.column_ids = None

    def gather_unordered(self, local: NDArray) -> NDArray | None:
        """
        Concatenate an array of every rank along its first axis on the root rank, in rank order.

        The arrays may differ in length along the first axis, but must have the same dtype and the same other
        dimensions on every rank.

        @param[in] local this rank's array

        @returns the concatenated array on the root rank, None on all other ranks
        """
        if self.comm.size == 1:
            return local

        local = np.ascontiguousarray(local)
        layouts = self.comm.gather((local.shape, local.dtype), root=self.root)

        if not self.is_root:
            self.comm.Gatherv(local, None, root=self.root)
            return None

        assert all(
            shape[1:] == local.shape[1:] and dtype == local.dtype for shape, dtype in layouts
        ), f"Cannot gather arrays of different dtypes or trailing dimensions (shape, dtype per rank): {layouts}"

        values_per_row = int(np.prod(local.shape[1:]))
        rows = [shape[0] for shape, _ in layouts]
        gathered = np.empty((sum(rows), *local.shape[1:]), dtype=local.dtype)
        self.comm.Gatherv(local, (gathered, [n * values_per_row for n in rows]), root=self.root)
        return gathered

    def gather(self, local: NDArray) -> NDArray | None:
        """
        Gather a per-column array on the root rank, ordered by column id.

        @param[in] local array whose first axis runs over this rank's columns, in the order of its column ids

        @returns the array for the columns of all ranks on the root rank, None on all other ranks
        """
        assert (
            local.shape[0] == self.n_local_columns
        ), f"Expected one row per column ({self.n_local_columns}), got an array of shape {local.shape}."

        if self.comm.size == 1:
            return local

        gathered = self.gather_unordered(local)
        if not self.is_root:
            return None
        return gathered[self._order]

    def scatter_unordered(self, values: NDArray | None) -> NDArray:
        """
        Split an array of the root rank along its first axis into consecutive blocks, one per rank in rank order.

        The inverse of gather_unordered: rank r receives as many rows as it has columns, the rows that follow those
        of the ranks before it.

        @param[in] values on the root rank, an array with one row per column of all ranks; ignored on other ranks

        @returns this rank's rows
        """
        if self.comm.size == 1:
            return values

        layout = None
        if self.is_root:
            values = np.ascontiguousarray(values)
            assert values.shape[0] == sum(
                self._counts
            ), f"Expected one row per column of all ranks ({sum(self._counts)}), got an array of shape {values.shape}."
            layout = (values.dtype, values.shape[1:])
        dtype, trailing_shape = self.comm.bcast(layout, root=self.root)

        local = np.empty((self.n_local_columns, *trailing_shape), dtype=dtype)
        values_per_row = int(np.prod(trailing_shape))
        send = (values, [n * values_per_row for n in self._counts]) if self.is_root else None
        self.comm.Scatterv(send, local, root=self.root)
        return local

    def scatter(self, values: NDArray | None) -> NDArray:
        """
        Give every rank the rows of its columns from an array of the root rank ordered by column id.

        The inverse of gather.

        @param[in] values on the root rank, an array with one row per column of all ranks, in the order of
                          column_ids; ignored on other ranks

        @returns the rows of this rank's columns, in the order of its column ids
        """
        if self.comm.size == 1:
            return values

        unordered = None
        if self.is_root:
            unordered = np.empty_like(values)
            unordered[self._order] = values
        return self.scatter_unordered(unordered)

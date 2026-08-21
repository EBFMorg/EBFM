# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Column discretization configuration for EBFM.

This file provides the configuration of how a single snow/firn column is divided into layers.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class ColumnDiscretizationConfig:
    """
    Column discretization configuration.

    Each glacier grid point carries one vertical snow/firn column of `nl` layers, from layer 0 (a thin surface ghost
    layer) down to the deepest layer.

    The number of columns (`gpsum`) is deliberately not configured here: it is derived from the loaded mesh in
    `init_grid`, so it belongs to the grid.

    With `doubledepth` enabled the layers get thicker with depth, doubling at each entry of `split`, so the deepest
    zone has layers of `2 ** len(split) * max_subZ`.

    `split` holds indices, which sit one layer below the start of each thickness zone.

    The defaults below are the model configuration; nothing overrides them from the command line. Tests pass explicit
    values to run smaller columns.
    """

    def __init__(
        self,
        nl: int = 50,  # Number of layers per column
        max_subZ: float = 0.1,  # Maximum thickness of the top layer (m)
        doubledepth: bool = True,  # Whether layer thickness doubles with depth
        split: Sequence[int] = (15, 25, 35),  # Merge trigger indices (see above)
    ) -> None:
        self.nl = nl
        self.max_subZ = max_subZ
        self.doubledepth = doubledepth
        self.split: NDArray[np.int_] = np.asarray(split)

        if self.nl < 3:
            # snowfall_and_deposition and melt_sublimation both treat layers 0 and 1 specially and shift the interior
            # between them, so a column needs at least one interior layer to be meaningful.
            raise ValueError(f"Column discretization requires nl >= 3, got nl={self.nl}.")

        if self.max_subZ <= 0.0:
            raise ValueError(f"Column discretization requires max_subZ > 0, got max_subZ={self.max_subZ}.")

        if self.split.ndim != 1 or self.split.size == 0:
            raise ValueError(f"Column discretization requires a non-empty 1-D split, got shape {self.split.shape}.")

        if np.any(np.diff(self.split) <= 0):
            raise ValueError(f"Column discretization requires a strictly increasing split, got {self.split.tolist()}.")

        # layer_merging_and_splitting indexes subZ[:, split] and subZ[:, split - 2], so every entry needs a layer on
        # both sides inside the column. Without the lower bound, split - 2 would wrap around to the deepest layer.
        if self.split[0] < 2:
            raise ValueError(f"Column discretization requires every split entry >= 2, got {self.split.tolist()}.")

        if self.split[-1] > self.nl - 1:
            raise ValueError(
                f"Column discretization requires every split entry <= nl - 1 = {self.nl - 1}, "
                f"got {self.split.tolist()}."
            )

# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

from coupling.couplers import Coupler

logger = logging.getLogger(__name__)


class OASISCoupler(Coupler):
    """
    Coupler implementation for OASIS coupling framework.
    """

    # TODO: Implement OASISCoupler methods
    # Serves as example for Strategy Pattern implementation of Coupler subclasses

    def __init__(self, coupling_config):
        super().__init__(coupling_config)
        raise NotImplementedError("OASISCoupler is not yet implemented.")

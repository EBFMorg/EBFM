# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections.abc import Collection, Mapping, Callable, Set as AbstractSet
import numpy as np

from ebfm.core import logging

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler
    from ebfm.coupling.fields.base import FieldSet


def identity(x: np.ndarray) -> np.ndarray:
    """
    Identity function for default transform.

    @param[in] x input data

    @returns the input data unchanged
    """
    return x


@dataclass(frozen=True)
class ExchangeKeySet:
    """
    The keys communicated in one exchange with a component: the keys to be put and the keys to be got.

    A component declares the key sets it accepts in Component.accepted_exchange_key_sets and compares the key
    set requested by the caller of exchange against them. A component that exchanges all of its data at once
    accepts a single key set, a component that requires its data to be exchanged in several calls (e.g. put the
    state in one call, get results derived from it in another call) accepts one key set per call. Note that a
    key set only defines which keys are communicated together, not in which order several key sets are
    exchanged: that is up to the caller and not enforced here.

    The keys are sets, so a set of keys declared by a component compares equal to the keys a caller asks for,
    independent of how either was created. The name is only used in log and error messages, hence two key sets
    with the same keys but different names are equal.
    """

    put_keys: AbstractSet[str] = frozenset()  # keys to be put, i.e. the keys of data_to_exchange
    get_keys: AbstractSet[str] = frozenset()  # keys to be got, i.e. the keys given by get_keys
    name: str = field(default="", compare=False)  # name of this key set, only used in messages


class Component(ABC):
    """
    Abstract base class for coupling components.
    Each component owns its fields as an instance attribute.
    """

    accepted_exchange_key_sets: tuple[ExchangeKeySet, ...] = ()
    """
    Key sets this component accepts in an exchange, i.e. which combinations of put and get keys a caller of
    exchange may ask for. Subclasses have to declare at least one key set.
    """

    def __init__(self, coupler: "Coupler", name: str):
        """
        Initialize the component.

        @param[in] coupler Coupler instance to use for coupling
        @param[in] name unique name of the component
        """
        self._coupler = coupler
        self.name = name
        pass

    def _uses_coupler(self, coupler_class_type) -> bool:
        """
        Check if the coupler is of a specific class type.

        This function is provided to avoid importing coupling libraries in component modules which would result in a
        circular dependency. You can check by providing the class name of the respective coupler

        Example: self._uses_coupler("YACCoupler")

        @param[in] coupler_class_type name of class type to check against

        @returns True if the coupler is of the specified class type, False otherwise
        """
        return self._coupler.__class__.__name__ == coupler_class_type

    def _put_if_coupled(
        self, field_name: str, data_to_exchange: Mapping[str, np.ndarray], transform: Callable = identity
    ):
        """
        Put a source field if it is coupled.

        @note Only call this for the put keys of the requested key set. The caller of exchange is required to
              provide the data for those keys, which exchange checks before anything is communicated.

        @param[in] field_name field name
        @param[in] data_to_exchange dictionary containing data to send
        @param[in] transform optional function to apply to the data before sending (e.g. for unit conversion)
        """
        from ebfm.coupling.fields.base import ExchangeType

        if self._coupler.has_field(self.name, field_name, ExchangeType.SOURCE):
            assert (
                field_name in data_to_exchange
            ), f"Field '{field_name}' is missing in data_to_exchange for component '{self.name}'."
            logger.debug(f"Putting data for field '{field_name}' to coupler: {data_to_exchange[field_name]}")
            err = self._coupler.put(self.name, field_name, transform(data_to_exchange[field_name]))
            if err:
                logger.warning(
                    f"Put for {field_name=} returned unexpected exit code ({err=}). "
                    "Please report this to the developers."
                )

    def _get_if_coupled(
        self, field_name: str, transform: Callable = identity, fallback_values: Mapping[str, np.ndarray] = {}
    ) -> np.ndarray | None:
        """
        Get a target field from the coupler if it is coupled.

        @note Only call this for the get keys of the exchange that is currently performed. Which keys those are
              is determined by the key set that the caller of exchange requested.

        @param[in] field_name field name
        @param[in] transform optional function to apply to the received data (e.g. for unit conversion)
        @param[in] fallback_values optional dictionary of fallback values to use if get fails

        @returns received field data if coupled, otherwise None
        """
        from ebfm.coupling.fields.base import ExchangeType

        if not self._coupler.has_field(self.name, field_name, ExchangeType.TARGET):
            logger.debug(f"Field '{field_name}' is not coupled for component '{self.name}', skipping get.")
            return None

        data, err = self._coupler.get(self.name, field_name)

        if err:
            logger.warning(f"Get for {field_name=} returned exit code ({err=}).")
            if field_name in fallback_values:
                logger.warning(
                    f"Using fallback value for '{field_name}' as no data was received for this field from {self.name}."
                )
                return fallback_values[field_name]
            return None

        assert data is not None, f"Received data for field '{field_name}' is None. {err}"
        logger.debug(f"Received data for field '{field_name}' from coupler: {data}")
        return transform(data)

    def _all_get_keys(self) -> set[str]:
        """
        Get all get keys of this component, i.e. the names of its coupled TARGET fields.

        These are the get keys a caller of exchange requests by default.

        @note Requires the coupler to be set up, since the coupled fields are only known after that.

        @returns names of the coupled TARGET fields of this component
        """
        from ebfm.coupling.fields.base import ExchangeType

        return self._coupler.get_field_names(self.name, ExchangeType.TARGET)

    def _requested_key_set(
        self, data_to_exchange: Mapping[str, np.ndarray], get_keys: Collection[str] | None
    ) -> ExchangeKeySet:
        """
        Build the key set requested by the caller of exchange, so that it can be compared to the key sets this
        component accepts (see accepted_exchange_key_sets).

        @param[in] data_to_exchange Mapping of field names to data to be sent, as given to exchange. Its keys are
                                    the put keys.
        @param[in] get_keys collection of field names to be received, as given to exchange. None is replaced by
                            all get keys of this component (see _all_get_keys).

        @returns the requested key set
        """
        requested_key_set = ExchangeKeySet(
            put_keys=frozenset(data_to_exchange),
            get_keys=frozenset(self._all_get_keys() if get_keys is None else get_keys),
        )
        logger.debug(f"Component '{self.name}' was asked to exchange {requested_key_set}.")
        return requested_key_set

    def _unsupported_exchange_message(self, requested_key_set: ExchangeKeySet) -> str:
        """
        Build the error message for a key set this component does not accept.

        Reports the requested keys, the key sets this component accepts and, for each of them, which keys are
        missing and which ones are not part of it.

        @param[in] requested_key_set key set requested by the caller of exchange

        @returns the error message
        """

        def _format_keys(keys: Collection[str]) -> str:
            """
            Format a set of field names for log and error messages.

            @param[in] keys field names to format

            @returns the field names in a deterministic (sorted) order
            """
            return "{" + ", ".join(f"'{key}'" for key in sorted(keys)) + "}"

        put_keys, get_keys = requested_key_set.put_keys, requested_key_set.get_keys

        lines = [
            f"Component '{self.name}' cannot exchange put keys {_format_keys(put_keys)} together with get keys "
            f"{_format_keys(get_keys)}.",
            "The put keys are the keys of data_to_exchange, the get keys are given by get_keys, which defaults "
            "to all get keys of this component.",
            "Accepted key sets of this component:",
        ]

        if not self.accepted_exchange_key_sets:
            lines.append("  none: this component does not declare any key set it accepts.")

        for accepted_key_set in self.accepted_exchange_key_sets:
            problems = []
            if missing := accepted_key_set.put_keys - put_keys:
                problems.append(f"missing put keys: {_format_keys(missing)}")
            if unexpected := put_keys - accepted_key_set.put_keys:
                problems.append(f"unexpected put keys: {_format_keys(unexpected)}")
            if missing := accepted_key_set.get_keys - get_keys:
                problems.append(f"missing get keys: {_format_keys(missing)}")
            if unexpected := get_keys - accepted_key_set.get_keys:
                problems.append(f"unexpected get keys: {_format_keys(unexpected)}")
            lines.append(
                f"  '{accepted_key_set.name}': put keys {_format_keys(accepted_key_set.put_keys)}, "
                f"get keys {_format_keys(accepted_key_set.get_keys)}"
            )
            if problems:
                lines.append(f"      {'; '.join(problems)}")

        return "\n".join(lines)

    def exchange(
        self,
        data_to_exchange: Mapping[str, np.ndarray],
        fallback_values: Mapping[str, np.ndarray] = {},
        get_keys: Collection[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Exchange of EBFM with this component

        The keys of `data_to_exchange` define what is sent (put keys), `get_keys` what is received (get keys).
        Both are unordered and together they have to be one of the key sets this component accepts (see
        accepted_exchange_key_sets), otherwise nothing is communicated and a ValueError is raised. The exchange
        itself is performed by _exchange. A component that cannot communicate all of its data in one call (e.g.
        because it derives what EBFM receives from what EBFM sends) accepts a key set per call and is called
        once per key set.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent
        @param[in] fallback_values optional Mapping of field names to fallback values to use if get fails
        @param[in] get_keys optional collection of field names to be received. Defaults to all get keys of this
                            component, i.e. the names of all its coupled TARGET fields.

        @returns dictionary of received field data. A requested field is not contained if it is not coupled, or
                 if no data was received for it and no fallback value was given.

        @raises ValueError if the keys of data_to_exchange and get_keys are not a key set this component accepts
        """
        requested_key_set = self._requested_key_set(data_to_exchange, get_keys)
        if requested_key_set not in self.accepted_exchange_key_sets:
            raise ValueError(self._unsupported_exchange_message(requested_key_set))

        return self._exchange(data_to_exchange, fallback_values, requested_key_set)

    @abstractmethod
    def _exchange(
        self,
        data_to_exchange: Mapping[str, np.ndarray],
        fallback_values: Mapping[str, np.ndarray],
        requested_key_set: ExchangeKeySet,
    ) -> dict[str, np.ndarray]:
        """
        Perform the exchange requested via exchange, which has already checked the requested key set.
        Subclasses must implement this method.

        Only put the put keys and only get the get keys of the requested key set. A component that accepts more
        than one key set therefore branches on requested_key_set.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent
        @param[in] fallback_values Mapping of field names to fallback values to use if get fails
        @param[in] requested_key_set key set to be communicated, one of accepted_exchange_key_sets

        @returns dictionary of received field data
        """
        pass

    @abstractmethod
    def get_field_definitions(self, time: dict[str, float]) -> "FieldSet":
        """
        Get field definitions for this component.
        Subclasses must implement this method.

        @param[in] time dictionary with time parameters
        @returns Set of Field objects for this component
        """
        pass

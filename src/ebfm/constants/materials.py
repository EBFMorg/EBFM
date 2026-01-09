# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final
from dataclasses import dataclass


@dataclass
class Material:
    DENSITY: Final[float]
    ALBEDO: float | None = None


@dataclass
class Ice(Material):
    DENSITY: Final[float] = 900.0
    ALBEDO: Final[float] = 0.39


@dataclass
class FreshSnow(Material):
    DENSITY: Final[float] = 350.0
    ALBEDO: Final[float] = 0.83


@dataclass
class Firn(Material):
    DENSITY: Final[float] = 500.0
    ALBEDO: Final[float] = 0.52


@dataclass
class Water(Material):
    DENSITY: Final[float] = 1000.0
    # no ALBEDO provided for Water

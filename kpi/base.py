"""
kpi/base.py

Abstract base class for all KPI formula implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseKPIFormula(ABC):
    """
    Contract for KPI formula implementations.

    Subclasses receive a plain dictionary of pre-fetched numerical inputs
    and must return a plain dictionary of computed metric values.

    No I/O, no logging, and no side effects are permitted inside
    :meth:`calculate`.
    """

    @abstractmethod
    def calculate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Compute KPI metrics from *inputs* and return a result dictionary.

        Parameters
        ----------
        inputs:
            Domain-specific numerical values required by the formula.

        Returns
        -------
        dict[str, Any]
            Computed metrics keyed by metric name.
        """

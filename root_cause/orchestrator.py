"""
root_cause/orchestrator.py

Routes root cause analysis requests to the appropriate engine
based on the declared business type.
"""

from __future__ import annotations

from root_cause.base import BaseRootCauseEngine
from root_cause.agency_rules import AgencyRootCauseEngine
from root_cause.ecommerce_rules import EcommerceRootCauseEngine
from root_cause.saas_rules import SaaSRootCauseEngine


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ENGINES: dict[str, BaseRootCauseEngine] = {
    "saas":       SaaSRootCauseEngine(),
    "ecommerce":  EcommerceRootCauseEngine(),
    "agency":     AgencyRootCauseEngine(),
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RootCauseOrchestrator:
    """
    Routes root cause analysis to the engine registered for a given
    business type.

    Each engine is instantiated once at import time and reused across
    calls.  Engines are stateless, so sharing instances is safe.

    Supported business types
    ------------------------
    - ``"saas"``       → :class:`~root_cause.saas_rules.SaaSRootCauseEngine`
    - ``"ecommerce"``  → :class:`~root_cause.ecommerce_rules.EcommerceRootCauseEngine`
    - ``"agency"``     → :class:`~root_cause.agency_rules.AgencyRootCauseEngine`
    """

    def analyze(
        self,
        business_type: str,
        kpi_data: dict,
        forecast_data: dict,
        risk_data: dict,
    ) -> dict:
        """
        Delegate root cause analysis to the matching engine.

        Parameters
        ----------
        business_type:
            Case-sensitive identifier for the business vertical.
            Must be one of the supported types listed in the class
            docstring.

        kpi_data:
            KPI metrics dictionary forwarded unchanged to the engine.

        forecast_data:
            Forecast result dictionary forwarded unchanged to the engine.

        risk_data:
            Risk assessment dictionary forwarded unchanged to the engine.

        Returns
        -------
        dict
            Result produced by the selected engine:
            ``{"primary_issue": str,
               "contributing_factors": List[str],
               "severity": str}``

        Raises
        ------
        ValueError
            If *business_type* is not registered.
        """
        engine: BaseRootCauseEngine | None = _ENGINES.get(business_type)

        if engine is None:
            supported = ", ".join(f'"{k}"' for k in _ENGINES)
            raise ValueError(
                f"Unsupported business_type '{business_type}'. "
                f"Supported values: {supported}."
            )

        return engine.analyze(kpi_data, forecast_data, risk_data)

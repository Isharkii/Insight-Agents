from __future__ import annotations

import math
from statistics import pstdev
from typing import Any

from kpi.base import BaseKPIFormula


class FinancialMarketsFormula(BaseKPIFormula):
    """
    Deterministic financial markets KPI formula set.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        pnl_series = _as_float_list(inputs.get("pnl_series", []))
        active_accounts = _as_int(inputs.get("active_accounts"))
        lost_accounts = _as_int(inputs.get("lost_accounts"))
        previous_revenue = _as_float(inputs.get("previous_revenue"))
        risk_free_rate = _as_optional_float(inputs.get("risk_free_rate"))
        volatility_override = _as_optional_float(inputs.get("volatility_override"))

        market_revenue = float(sum(pnl_series))
        account_churn = _safe_divide(float(lost_accounts), float(active_accounts))
        growth_rate = _safe_growth(market_revenue, previous_revenue)

        if volatility_override is not None:
            volatility = volatility_override
        elif len(pnl_series) >= 2:
            volatility = float(pstdev(pnl_series))
        else:
            volatility = None

        sharpe_like = _safe_sharpe(growth_rate, risk_free_rate, volatility)

        return {
            "market_revenue": market_revenue,
            "account_churn": account_churn,
            "growth_rate": growth_rate,
            "volatility": volatility,
            "sharpe_like": sharpe_like,
        }


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _safe_growth(current: float, previous: float) -> float | None:
    if previous == 0.0:
        return None
    return (current - previous) / previous


def _safe_sharpe(
    growth_rate: float | None,
    risk_free_rate: float | None,
    volatility: float | None,
) -> float | None:
    if growth_rate is None or risk_free_rate is None:
        return None
    if volatility is None or volatility == 0.0:
        return None
    return (growth_rate - risk_free_rate) / volatility


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        number = _as_optional_float(item)
        if number is not None and math.isfinite(number):
            out.append(float(number))
    return out


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_float(value: Any) -> float:
    parsed = _as_optional_float(value)
    return parsed if parsed is not None else 0.0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

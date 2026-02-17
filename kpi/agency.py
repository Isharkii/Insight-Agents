"""
kpi/agency.py

Marketing agency KPI formula implementation.

Expected inputs
---------------
retainer_fees : list[float]
    Monthly retainer amount for each active client in the period.
project_values : list[float]
    One-time project revenue amounts billed in the period.
starting_clients : int
    Number of active clients at the beginning of the period.
lost_clients : int
    Number of clients who cancelled or lapsed during the period.
billable_hours : float
    Total hours billed to clients in the period.
available_hours : float
    Total hours the team was available to work in the period.
total_employees : int
    Headcount used for per-employee revenue calculation.
average_client_lifespan_months : int
    Expected or observed average months a client stays.

Formulas
--------
Retainer Revenue    = sum(retainer_fees)
Project Revenue     = sum(project_values)
Total Revenue       = retainer_revenue + project_revenue
Client Churn        = lost_clients / starting_clients
Utilization Rate    = billable_hours / available_hours
Revenue Per Employee= total_revenue / total_employees
Average Retainer    = retainer_revenue / number_of_retainer_clients
Client LTV          = average_retainer * average_client_lifespan_months

Division-by-zero cases return None for the affected metric.
"""

from __future__ import annotations

from typing import Any

from kpi.base import BaseKPIFormula

_SENTINEL = None  # value stored when a metric cannot be computed


class AgencyKPIFormula(BaseKPIFormula):
    """
    Deterministic marketing agency KPI calculations with safe division-by-zero handling.

    All arithmetic is self-contained.  No I/O, no logging, no side effects.
    """

    def calculate(self, inputs: dict[str, Any]) -> dict[str, float | None]:
        """
        Compute retainer revenue, project revenue, total revenue, client churn,
        utilization rate, revenue per employee, and client LTV from *inputs*.

        Parameters
        ----------
        inputs:
            Dictionary containing the keys listed in the module docstring.

        Returns
        -------
        dict
            Keys: ``retainer_revenue``, ``project_revenue``, ``total_revenue``,
            ``client_churn``, ``utilization_rate``, ``revenue_per_employee``,
            ``client_ltv``.
            A metric is ``None`` when its formula produces a division by zero.
        """
        retainer_fees: list[float] = inputs["retainer_fees"]
        project_values: list[float] = inputs["project_values"]
        starting_clients: int = inputs["starting_clients"]
        lost_clients: int = inputs["lost_clients"]
        billable_hours: float = inputs["billable_hours"]
        available_hours: float = inputs["available_hours"]
        total_employees: int = inputs["total_employees"]
        average_client_lifespan_months: int = inputs["average_client_lifespan_months"]

        retainer_revenue = _retainer_revenue(retainer_fees)
        project_revenue = _project_revenue(project_values)
        total_revenue = _total_revenue(retainer_revenue, project_revenue)
        client_churn = _client_churn(lost_clients, starting_clients)
        utilization_rate = _utilization_rate(billable_hours, available_hours)
        revenue_per_employee = _revenue_per_employee(total_revenue, total_employees)
        average_retainer = _average_retainer(retainer_revenue, len(retainer_fees))
        client_ltv = _client_ltv(average_retainer, average_client_lifespan_months)

        return {
            "retainer_revenue": retainer_revenue,
            "project_revenue": project_revenue,
            "total_revenue": total_revenue,
            "client_churn": client_churn,
            "utilization_rate": utilization_rate,
            "revenue_per_employee": revenue_per_employee,
            "client_ltv": client_ltv,
        }


# ---------------------------------------------------------------------------
# Pure formula functions
# ---------------------------------------------------------------------------


def _retainer_revenue(retainer_fees: list[float]) -> float:
    """Retainer Revenue = sum of all monthly retainer fees."""
    return sum(retainer_fees)


def _project_revenue(project_values: list[float]) -> float:
    """Project Revenue = sum of all one-time project billings."""
    return sum(project_values)


def _total_revenue(retainer_revenue: float, project_revenue: float) -> float:
    """Total Revenue = retainer_revenue + project_revenue."""
    return retainer_revenue + project_revenue


def _client_churn(lost_clients: int, starting_clients: int) -> float | None:
    """
    Client Churn = lost_clients / starting_clients.

    Returns None when starting_clients is zero.
    """
    if starting_clients == 0:
        return _SENTINEL
    return lost_clients / starting_clients


def _utilization_rate(billable_hours: float, available_hours: float) -> float | None:
    """
    Utilization Rate = billable_hours / available_hours.

    Returns None when available_hours is zero.
    """
    if available_hours == 0.0:
        return _SENTINEL
    return billable_hours / available_hours


def _revenue_per_employee(total_revenue: float, total_employees: int) -> float | None:
    """
    Revenue Per Employee = total_revenue / total_employees.

    Returns None when total_employees is zero.
    """
    if total_employees == 0:
        return _SENTINEL
    return total_revenue / total_employees


def _average_retainer(retainer_revenue: float, number_of_retainer_clients: int) -> float | None:
    """
    Average Retainer = retainer_revenue / number_of_retainer_clients.

    Returns None when number_of_retainer_clients is zero.
    """
    if number_of_retainer_clients == 0:
        return _SENTINEL
    return retainer_revenue / number_of_retainer_clients


def _client_ltv(
    average_retainer: float | None,
    average_client_lifespan_months: int,
) -> float | None:
    """
    Client LTV = average_retainer * average_client_lifespan_months.

    Returns None when average_retainer is None (upstream division by zero).
    """
    if average_retainer is None:
        return _SENTINEL
    return average_retainer * average_client_lifespan_months

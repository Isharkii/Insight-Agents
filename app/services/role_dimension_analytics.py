from __future__ import annotations

import math
from typing import Any, Mapping


SUPPORTED_ROLE_DIMENSIONS: tuple[str, ...] = (
    "team",
    "channel",
    "region",
    "product_line",
)


def build_role_dimension_summary(
    rows: list[Mapping[str, Any]],
    *,
    top_n: int = 3,
    dimensions: tuple[str, ...] = SUPPORTED_ROLE_DIMENSIONS,
) -> dict[str, Any]:
    """
    Build deterministic contribution analytics across supported dimensions.
    """
    normalized_rows = [_normalize_row(row) for row in rows]
    used_rows = [row for row in normalized_rows if row.get("metric_value") is not None]

    by_dimension: dict[str, dict[str, Any]] = {}
    flattened: list[dict[str, Any]] = []

    for dimension in dimensions:
        groups: dict[str, dict[str, float | int]] = {}
        for row in used_rows:
            group_name = _dimension_value(row, dimension)
            if not group_name:
                continue
            metric_value = row["metric_value"]
            if metric_value is None:
                continue
            group = groups.setdefault(group_name, {"contribution_value": 0.0, "rows": 0})
            group["contribution_value"] = float(group["contribution_value"]) + float(metric_value)
            group["rows"] = int(group["rows"]) + 1

        contributors = _contributors_for_dimension(dimension, groups)
        top_contributors = contributors[:top_n]
        laggards = sorted(
            contributors,
            key=lambda item: (item["contribution_share"], item["contribution_value"], item["name"]),
        )[:top_n]
        concentration = _dependency_concentration(contributors)

        by_dimension[dimension] = {
            "contributors": contributors,
            "top_contributors": top_contributors,
            "laggards": laggards,
            "dependency_concentration": concentration,
            "total_contribution": round(sum(item["contribution_value"] for item in contributors), 6),
            "group_count": len(contributors),
        }
        flattened.extend(top_contributors)

    top_contributors_global = sorted(
        flattened,
        key=lambda item: (-item["contribution_share"], -item["contribution_value"], item["dimension"], item["name"]),
    )[:top_n]
    laggards_global = sorted(
        (
            contributor
            for payload in by_dimension.values()
            for contributor in payload.get("contributors", [])
        ),
        key=lambda item: (item["contribution_share"], item["contribution_value"], item["dimension"], item["name"]),
    )[:top_n]

    concentration_by_dimension = {
        dimension: payload["dependency_concentration"]
        for dimension, payload in by_dimension.items()
    }
    most_concentrated = _pick_dimension_extreme(concentration_by_dimension, highest=True)
    least_concentrated = _pick_dimension_extreme(concentration_by_dimension, highest=False)

    return {
        "dimensions": list(dimensions),
        "records_scanned": len(normalized_rows),
        "records_used": len(used_rows),
        "top_contributors": top_contributors_global,
        "laggards": laggards_global,
        "dependency_concentration": {
            "by_dimension": concentration_by_dimension,
            "most_concentrated_dimension": most_concentrated,
            "least_concentrated_dimension": least_concentrated,
        },
        "by_dimension": by_dimension,
    }


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata_json")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "team": row.get("team"),
        "channel": row.get("channel"),
        "region": row.get("region"),
        "product_line": row.get("product_line"),
        "role": row.get("role"),
        "source_type": row.get("source_type"),
        "metadata_json": dict(metadata),
        "metric_value": _coerce_float(row.get("metric_value")),
    }


def _dimension_value(row: Mapping[str, Any], dimension: str) -> str | None:
    metadata = row.get("metadata_json")
    if not isinstance(metadata, Mapping):
        metadata = {}

    if dimension == "team":
        raw = row.get("team") or row.get("role") or metadata.get("team") or metadata.get("team_name")
    elif dimension == "channel":
        raw = row.get("channel") or metadata.get("channel") or metadata.get("source_channel") or row.get("source_type")
    elif dimension == "region":
        raw = row.get("region") or metadata.get("region") or metadata.get("market")
    elif dimension == "product_line":
        raw = (
            row.get("product_line")
            or metadata.get("product_line")
            or metadata.get("product")
            or metadata.get("service_line")
        )
    else:
        raw = row.get(dimension) or metadata.get(dimension)

    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    return value


def _contributors_for_dimension(
    dimension: str,
    groups: Mapping[str, Mapping[str, float | int]],
) -> list[dict[str, Any]]:
    total = sum(max(0.0, float(payload.get("contribution_value") or 0.0)) for payload in groups.values())
    contributors: list[dict[str, Any]] = []
    for name, payload in groups.items():
        contribution_value = float(payload.get("contribution_value") or 0.0)
        effective_value = max(0.0, contribution_value)
        share = (effective_value / total) if total > 0.0 else 0.0
        contributors.append(
            {
                "dimension": dimension,
                "name": str(name),
                "contribution_value": round(contribution_value, 6),
                "contribution_share": round(share, 6),
                "rows": int(payload.get("rows") or 0),
            }
        )
    contributors.sort(
        key=lambda item: (-item["contribution_share"], -item["contribution_value"], item["name"])
    )
    return contributors


def _dependency_concentration(contributors: list[Mapping[str, Any]]) -> dict[str, Any]:
    shares = [float(item.get("contribution_share") or 0.0) for item in contributors]
    hhi = round(sum(share * share for share in shares), 6)
    dominant = contributors[0] if contributors else None
    effective_groups = round((1.0 / hhi), 6) if hhi > 0.0 else 0.0
    return {
        "hhi": hhi,
        "effective_groups": effective_groups,
        "dominant_contributor": dominant,
    }


def _pick_dimension_extreme(
    concentration_by_dimension: Mapping[str, Mapping[str, Any]],
    *,
    highest: bool,
) -> dict[str, Any] | None:
    if not concentration_by_dimension:
        return None
    ordered = sorted(
        (
            {
                "dimension": dimension,
                "hhi": float(payload.get("hhi") or 0.0),
                "effective_groups": float(payload.get("effective_groups") or 0.0),
                "dominant_contributor": payload.get("dominant_contributor"),
            }
            for dimension, payload in concentration_by_dimension.items()
        ),
        key=lambda item: (item["hhi"], item["dimension"]),
        reverse=highest,
    )
    return ordered[0] if ordered else None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed

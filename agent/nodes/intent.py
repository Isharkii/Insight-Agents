"""
agent/nodes/intent.py

Intent node: extracts business_type and entity_name from user_query
using deterministic keyword rules. No LLM, no DB.
"""

import re
from agent.state import AgentState


# ---------------------------------------------------------------------------
# Keyword maps
# ---------------------------------------------------------------------------

BUSINESS_TYPE_KEYWORDS: dict[str, list[str]] = {
    "retail": ["retail", "store", "shop", "ecommerce", "e-commerce", "merchandise"],
    "food_service": ["restaurant", "cafe", "cafeteria", "diner", "food", "beverage", "catering"],
    "saas": ["saas", "software", "platform", "app", "subscription", "cloud service"],
    "finance": ["bank", "finance", "fintech", "investment", "insurance", "lending", "credit"],
    "healthcare": ["hospital", "clinic", "healthcare", "pharma", "medical", "pharmacy", "dental"],
    "logistics": ["logistics", "shipping", "delivery", "supply chain", "warehouse", "freight"],
    "real_estate": ["real estate", "property", "realty", "housing", "apartment", "rental"],
    "education": ["school", "university", "college", "education", "edtech", "tutoring", "course"],
    "manufacturing": ["manufacturing", "factory", "production", "assembly", "plant", "industrial"],
    "hospitality": ["hotel", "hospitality", "resort", "travel", "tourism", "airline"],
}

# Patterns that typically precede an entity/brand name in a query
_ENTITY_PATTERNS: list[str] = [
    r"for\s+([A-Z][A-Za-z0-9&'\-\s]{1,40}?)(?:\s+(?:in|at|on|is|are|has|have|with|,|$))",
    r"of\s+([A-Z][A-Za-z0-9&'\-\s]{1,40}?)(?:\s+(?:in|at|on|is|are|has|have|with|,|$))",
    r"about\s+([A-Z][A-Za-z0-9&'\-\s]{1,40}?)(?:\s+(?:in|at|on|is|are|has|have|with|,|$))",
    r"(?:analyze|analyse|evaluate|assess|review)\s+([A-Z][A-Za-z0-9&'\-\s]{1,40}?)(?:\s+(?:in|at|on|is|are|has|have|with|,|$|\.))",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_business_type(query: str) -> str:
    """Return the best-matching business type or 'general' if none found."""
    lowered = query.lower()
    scores: dict[str, int] = {}

    for btype, keywords in BUSINESS_TYPE_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw in lowered)
        if hit:
            scores[btype] = hit

    if not scores:
        return "general"

    return max(scores, key=lambda k: scores[k])


def _extract_entity_name(query: str) -> str:
    """
    Attempt to pull a proper-noun entity name from the query.
    Returns empty string if nothing plausible is found.
    """
    for pattern in _ENTITY_PATTERNS:
        match = re.search(pattern, query)
        if match:
            candidate = match.group(1).strip(" ,.")
            if candidate:
                return candidate

    # Fallback: look for runs of Title-Cased words (2–4 tokens)
    tokens = query.split()
    title_run: list[str] = []
    for token in tokens:
        clean = token.strip(",.!?()")
        if clean and clean[0].isupper() and not clean.isupper():
            title_run.append(clean)
        else:
            if len(title_run) >= 2:
                break
            title_run = []

    if len(title_run) >= 2:
        return " ".join(title_run)

    return ""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def intent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: parse user_query to populate business_type and entity_name.

    Updates state fields:
        - business_type  (overwritten only if still empty / 'general')
        - entity_name    (overwritten only if still empty)

    All other state fields are left untouched.
    """
    query: str = state.get("user_query", "").strip()

    detected_type = _detect_business_type(query)
    detected_entity = _extract_entity_name(query)

    # Only overwrite if the caller left the field blank or at default
    new_business_type = (
        detected_type
        if not state.get("business_type") or state["business_type"] == "general"
        else state["business_type"]
    )

    new_entity_name = (
        detected_entity
        if not state.get("entity_name")
        else state["entity_name"]
    )

    return {
        **state,
        "business_type": new_business_type,
        "entity_name": new_entity_name,
    }

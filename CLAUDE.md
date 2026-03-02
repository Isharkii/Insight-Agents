Project: Deterministic Insight & Competitive Intelligence Agent

(Lightweight, Local-First, Confidence-Governed)

Core Philosophy

This system is a deterministic-first intelligence engine.

LLMs do not decide facts.
LLMs explain validated signals.

The architecture must:

Be lightweight

Run locally

Scale to cloud

Enforce data integrity

Prevent misleading narratives

Preserve auditability

Remain vendor-agnostic

This is not an orchestration tool.

This is a portable intelligence substrate.

Primary Objective

Build a modular AI Insight & Benchmarking Agent that:

Processes structured business data

Computes KPIs deterministically

Enforces temporal integrity (no double counting)

Propagates confidence mathematically

Performs direction-aware competitive benchmarking

Generates confidence-aware narratives

Outputs strictly structured JSON

Runs locally

Scales via API in cloud mode

Non-Negotiable Design Principles
1. Deterministic Before Generative

All calculations must occur before LLM synthesis:

KPI computation

Growth rates

Volatility

Trend detection

Benchmark ranking

Confidence scoring

Risk categorization

The LLM may:

Explain

Reframe

Strategize

The LLM must not:

Compute metrics

Infer missing numbers

Decide rankings

Override deterministic confidence

2. Confidence-Governed Synthesis

Synthesis is allowed only if:

deterministic_confidence >= MIN_CONFIDENCE_FOR_SYNTHESIS

Tone must align with confidence:

≥ 0.8 → definitive

0.6–0.8 → cautious

0.4–0.6 → hedged

< 0.4 → synthesis blocked

Confidence must propagate from:

KPI → Risk → Diagnostic → Enforcement → Output

No narrative without statistical substrate.

3. Temporal Integrity Contract

All time-series windows must use:

[start, end) half-open intervals

Rules:

No double counting

No boundary overlap

Final month must not be dropped

Same window alignment across entities

Temporal correctness is foundational.

4. Metric Provenance Required

Each KPI must carry:

{
"value": float,
"unit": string,
"source": "formula" | "precomputed_backfill",
"window": { "start": "", "end": "" }
}

No silent fallback.
No hidden degradation.

Provenance must propagate to:

LLM

Dashboard

Exports

5. Competitive Benchmarking Rules

Benchmarking must:

Compare across peer entities (never self-only)

Align time windows

Align aggregation level

Align units and currency

Normalize scale (% vs fraction)

Respect metric direction (lower_is_better supported)

Incorporate volatility/stability

Optionally weight by confidence

Ranking must be deterministic and reproducible.

Lower-is-better metrics (e.g., churn) must invert scoring.

No float-only comparison without metadata validation.

System Layers (Updated)
1. Data Layer

CSV / Excel ingestion

Clean + normalize

Enforce timestamp alignment

Enforce entity isolation

Memory-efficient processing

2. Insight Layer

KPI formulas (deterministic)

Growth %

Trend slope

Volatility score

Stability index

Confidence penalties

Window validation

No ML unless justified.

3. Benchmark Layer

Peer retrieval

MetricComparisonSpec validation

Direction-aware scoring

Composite scoring

Market positioning classification:

Leader

Challenger

Stable

Declining

4. Reasoning Layer

Structured prompt builder

Confidence-aware tone control

Missing signal injection

Model-agnostic interface

LLM must receive:

Deterministic diagnostics

Confidence score

Missing signals

Competitive deltas

5. Strategy Layer

Convert diagnostics into:

Immediate actions

Defensive strategies

Offensive strategies

Risk framing

Opportunity classification

Strategy must reference metrics explicitly.

Model Abstraction (Required)

All models must implement:

class BaseModelInterface:
    def generate(self, prompt: str) -> dict:
        pass

No vendor lock-in.
No provider-specific logic in business code.

Output Contract (Strict)

All responses must follow structured schema.

Example (Insight Mode):

{
"insight_summary": "",
"risk_assessment": "",
"market_position": "",
"evidence": [],
"recommended_actions": {
"immediate": [],
"mid_term": [],
"defensive": [],
"offensive": []
},
"confidence_score": float,
"confidence_drivers": {
"missing_signals": [],
"degraded_signals": []
}
}

No unstructured text.

Performance Constraints

Must run on local laptop

Avoid large memory spikes

Avoid unnecessary vector DB

Avoid heavy orchestration

Lazy load when possible

Cloud mode optional via FastAPI

What This System Is NOT

Not a chatbot

Not n8n-style workflow automation

Not narrative-first AI

Not metric computation inside LLM

Not dependent on a single vendor

Not a UI-only analytics tool

Long-Term Direction

Evolve into:

Confidence-aware business intelligence core

Competitive strategy engine

Embedded dashboard intelligence

Pitch-ready insight generator (PitchWorx use case)

Multi-tenant cloud deployable API

Audit-safe analytics substrate

Decision Priority Order (Updated)

When making architectural decisions, prioritize:

Deterministic correctness

Temporal integrity

Confidence governance

Benchmark validity

Structured outputs

Lightweight execution

Cloud compatibility

Sellable product design

Engineering Rule

If a feature increases narrative sophistication
but weakens mathematical integrity,
reject it.

If a feature increases correctness
but slightly increases complexity,
accept it.
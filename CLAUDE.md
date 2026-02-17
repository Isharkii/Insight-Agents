# Project: Lightweight Insight Agent (Local-First, Cloud-Scalable)

## Core Philosophy

This project is designed to be:

- Lightweight
- Modular
- Locally runnable
- Resource-efficient
- Capable of handling complex reasoning
- Deployable to cloud when needed

This is NOT a heavy enterprise monolith.
This is a portable intelligence core.

It must run locally on modest hardware,
but scale when deployed to cloud infrastructure.

---

# Primary Goal

Build a minimal, modular AI Insight Agent that:

1. Processes structured data
2. Generates meaningful insights
3. Handles complex prompts
4. Produces strategic recommendations
5. Outputs structured responses
6. Can run locally
7. Can scale via cloud deployment

---

# Architectural Constraints

## Default Mode: Local Execution

- Minimal dependencies
- No unnecessary vector DB
- No heavy orchestration frameworks unless needed
- Prefer simple pipelines over distributed systems
- Avoid microservice over-engineering

## Scalable Mode: Cloud Deployment

When deployed to cloud, system should:

- Support API wrapping (FastAPI)
- Handle larger datasets
- Enable remote LLM calls
- Support multi-client configuration
- Add vector search if required

Architecture must allow switching between:

LOCAL MODE
CLOUD MODE

Without rewriting core logic.

---

# System Layers

## 1. Data Layer
- CSV / Excel ingestion
- API ingestion
- Optional web scraping
- Clean + normalize data
- Memory efficient

## 2. Insight Layer
- Statistical metrics
- Trend detection
- Anomaly detection
- Pattern analysis
- KPI computation

Keep this lightweight.
Use Pandas, NumPy.
Avoid unnecessary ML unless needed.

## 3. Reasoning Layer

Must handle complex prompts.

- Structured prompt templates
- Chain-of-thought internally
- JSON structured output
- Model-agnostic API wrapper
- Local LLM compatible (optional)
- Cloud LLM compatible

Never hardcode vendor-specific logic.

## 4. Strategy Layer

Convert insights into:
- Actionable recommendations
- Risk evaluation
- Opportunity scoring
- Executive summaries

---

# Key Requirement: Lightweight but Capable

The system must:

- Run on local laptop
- Handle complex reasoning prompts
- Avoid large memory footprint
- Avoid heavy GPU requirements (unless optional)
- Allow cloud offloading if prompt complexity increases

---

# Model Strategy

Support:

- Local small models (if configured)
- API-based LLMs
- Switchable model backend

Create a model abstraction layer:

class BaseModelInterface:
    def generate(prompt: str) -> dict:
        pass

All model implementations must follow this interface.

---

# Output Format (Always Structured)

{
  "insight": "",
  "evidence": "",
  "impact": "",
  "recommended_action": "",
  "priority": "",
  "confidence_score": ""
}

Never return unstructured paragraphs unless explicitly requested.

---

# Coding Rules

- Keep modules small
- Use typing
- Add docstrings
- Keep logic separated:
    - data
    - insights
    - reasoning
    - strategy
- No heavy frameworks unless justified
- No unnecessary abstractions
- No hardcoded business rules

---

# Performance Principles

- Optimize for low memory usage
- Avoid loading full datasets if streaming is possible
- Lazy load components when needed
- Avoid long blocking calls

---

# Prompt Handling Rules

When processing complex prompts:

- Break down internally
- Reason step-by-step
- Summarize efficiently
- Avoid hallucinated assumptions
- Provide explainable output

---

# What This Is NOT

- Not a chatbot
- Not a dashboard-only tool
- Not a heavy AI research system
- Not tied to one LLM provider

---

# Long-Term Direction

Evolve into:

- Lightweight AI engine
- Cloud-deployable insight core
- Embedded intelligence for dashboards
- PPT insight generator (PitchWorx use case)
- Configurable per client domain

---

# Decision Priority Order

If unsure, prioritize:

1. Lightweight architecture
2. Modularity
3. Structured output
4. Cloud compatibility
5. Clear reasoning
6. Sellable product design

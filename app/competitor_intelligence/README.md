# Competitor Intelligence Module

Deterministic-first module for SaaS competitor benchmarking with strict structured outputs.

## Environment Variables

- `COMP_INTEL_PROVIDER` = `brave | serper | tavily`
- `COMP_INTEL_BRAVE_API_KEY`
- `COMP_INTEL_SERPER_API_KEY`
- `COMP_INTEL_TAVILY_API_KEY`
- `COMP_INTEL_CACHE_ENABLED` = `true | false`
- `COMP_INTEL_CACHE_TTL_SECONDS`
- `COMP_INTEL_EXTRACTION_MODE` = `deterministic | llm_structured`
- `COMP_INTEL_LLM_API_KEY` (required for `llm_structured`)
- `COMP_INTEL_LLM_MODEL` (default `gpt-5.4`)

## Quick Usage

```python
import asyncio

from app.competitor_intelligence import build_competitor_intelligence_service
from app.competitor_intelligence.schemas import CompetitorIntelligenceRequest


async def main() -> None:
    service = build_competitor_intelligence_service()
    result = await service.generate(
        CompetitorIntelligenceRequest(
            subject_entity="Acme SaaS",
            competitors=["Contoso", "Globex"],
        )
    )
    print(result.model_dump_json(indent=2))


asyncio.run(main())
```

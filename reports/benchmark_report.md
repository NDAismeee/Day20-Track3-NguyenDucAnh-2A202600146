# Benchmark Report

| Run | Latency (s) | Tokens (in/out) | Citation coverage | Success | Notes |
|---|---:|---:|---:|---:|---|
| baseline:q1_graphrag | 25.06 | 65/569 | 0.00 | 1 | quality_score is heuristic; replace with peer review 0-10 |
| multi-agent:q1_graphrag | 31.75 | 2105/1508 | 0.50 | 1 | quality_score is heuristic; replace with peer review 0-10 |
| baseline:q2_langgraph | 8.43 | 64/348 | 0.00 | 1 | quality_score is heuristic; replace with peer review 0-10 |
| multi-agent:q2_langgraph | 36.34 | 2088/1454 | 0.62 | 1 | quality_score is heuristic; replace with peer review 0-10 |
| baseline:q3_failure_modes | 8.13 | 62/368 | 0.00 | 1 | quality_score is heuristic; replace with peer review 0-10 |
| multi-agent:q3_failure_modes | 30.37 | 2183/1712 | 0.36 | 1 | quality_score is heuristic; replace with peer review 0-10 |

## Interpretation

- Average latency (baseline): **13.87s**
- Average latency (multi-agent): **32.82s**
- Average citation coverage (baseline): **0.00** (expected low/no citations)
- Average citation coverage (multi-agent): **0.50**
- Failure rate (all runs): **0.00%**

## Failure modes observed / expected

- Search returns no sources (or low-quality sources) → weak grounding and low citation coverage.
- Tool/API timeouts → workflow returns partial state; mitigate with retries/fallback and smaller prompts.
- Overlong intermediate notes → higher latency/cost; mitigate with strict output constraints per agent.

## Notes

- `quality_score` in this report is **heuristic**; replace with peer review score (0–10) using the rubric.
- Traces are exported under `reports/traces/` for each multi-agent run.

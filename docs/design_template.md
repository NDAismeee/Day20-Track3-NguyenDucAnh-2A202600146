# Design Template

## Problem

Build a research assistant that can take a long, ambiguous technical query and produce a grounded final answer by:

- collecting sources (URLs/snippets)
- synthesizing research notes with citations
- producing analysis (claims/pros/cons/gaps)
- writing a final response that keeps citations and highlights what to verify next

The system must support both:

- a **single-agent baseline** (one LLM call)
- a **multi-agent workflow** coordinated by a Supervisor (Researcher → Analyst → Writer)

## Why multi-agent?

Single-agent often mixes searching, reasoning, and writing in one step, which makes it harder to:

- keep a clear separation between **grounding (sources)** vs **analysis** vs **final writing**
- debug where quality failed (search vs reasoning vs writing)
- enforce guardrails per stage (timeouts/size constraints)
- benchmark and attribute cost/latency to specific stages

Multi-agent improves **role clarity**, makes shared state explicit for handoffs, and enables routing/fallback when one stage fails.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Decide next step and stop condition based on missing fields and failures | `ResearchState` (iteration, notes present/missing, failure_counts) | `route_history` updated with next route (`researcher` / `analyst` / `writer` / `done`) | Infinite loop risk if stop conditions missing; mitigated by `MAX_ITERATIONS` + recursion limit |
| Researcher | Search and summarize sources into concise notes with citations | query + max_sources + audience | `sources[]`, `research_notes` (bullets ending with `[n]`) | No sources / low-quality sources; search API failure; hallucinated citations |
| Analyst | Convert research into structured analysis and identify gaps | `research_notes` (+ optional sources index) | `analysis_notes` (claims, pros/cons, reasoning, gaps) | Weak reasoning, missing gaps, loses citation markers |
| Writer | Write final answer grounded in notes with citations and “what to verify next” | query + audience + sources index + research_notes + analysis_notes | `final_answer` (validated non-empty, citations `[n]`) | Empty answer, missing citations, overlong output |

## Shared state

`ResearchState` is the single source of truth that supports debug, tracing, evaluation, and safe handoffs.

- **request** (`ResearchQuery`): query, max_sources, audience (inputs should be schema validated)
- **messages** (`ChatMessage[]`): optional conversation/context history for future extensions
- **iteration**: current iteration counter (for stop/guardrails)
- **route_history**: list of routes chosen by Supervisor (for trace explanation)
- **failure_counts**: per-agent failure counters (for retry/fallback decisions)
- **sources** (`SourceDocument[]`): retrieved sources with `title/url/snippet` (grounding + citations)
- **research_notes**: researcher output used as input to analyst/writer
- **analysis_notes**: analyst output used as input to writer
- **final_answer**: final user-facing answer (must be non-empty)
- **usage** (`UsageMetrics`): aggregated token usage and optional cost
- **timing** (`TimingMetrics`): per-stage seconds and optional wall time
- **trace**: structured events/spans for screenshot/export
- **errors**: human-readable errors for failure explanation and benchmark failure-rate

## Routing policy

Target graph (implemented with LangGraph):

```text
supervisor
   |--(missing research or sources)--> researcher --> supervisor
   |--(missing analysis)-------------> analyst ----> supervisor
   |--(ready to write)--------------> writer -----> supervisor
   |--(final_answer present)--------> done
   |--(max_iterations exceeded)-----> done
```

Failure-aware routing:

- Each node records failures into `failure_counts` and `errors`
- If a route fails repeatedly (>=2), Supervisor falls back to `writer` (best-effort) or stops if writer also fails too many times

## Guardrails

- Max iterations: `MAX_ITERATIONS` enforced in Supervisor (stop to `done` when exceeded) + LangGraph `recursion_limit`
- Timeout: `TIMEOUT_SECONDS` enforced at workflow execution level (hard timeout returns partial state)
- Retry: LLM calls retry up to 3 times (exponential backoff) inside `LLMClient.complete()`
- Fallback:
  - Search: use Tavily if `TAVILY_API_KEY` else local mock corpus (`mock_data/search_corpus.json`)
  - Routing: if an agent fails >=2 times, Supervisor falls back to writer or stops (avoids infinite retries)
- Validation:
  - schema-level: `min_length=1` for message/agent outputs and snippets
  - runtime: `set_final_answer()` raises if blank; agents raise if notes are empty; negative timings rejected

## Benchmark plan

Query set (implemented in `mock_data/benchmark_queries.json`):

- `q1_graphrag`: explain GraphRAG vs RAG + implementation checklist
- `q2_langgraph`: summarize LangGraph concepts + when multi-agent is justified
- `q3_failure_modes`: list failure modes + guardrails

Metrics collected (baseline vs multi-agent):

- Latency: wall-clock seconds per run
- Cost: token usage (input/output tokens); USD cost if provider exposes it
- Quality score: placeholder heuristic in code; replace with peer rubric score (0–10)
- Citation coverage: fraction of bullet/sentence units containing citations like `[1]`
- Failure rate: percent of runs that fail/return empty outputs

Expected outcome:

- Multi-agent has higher latency and token usage (more steps), but higher citation coverage and more structured answers.

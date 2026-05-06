"""Benchmark skeleton for single-agent vs multi-agent."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]

_CITE_RE = re.compile(r"\[\d+\]")


def _citation_coverage(text: str | None) -> float | None:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    units: list[str]
    bulletish = [ln for ln in lines if ln.startswith(("-", "*")) or ln[:2].isdigit()]
    if bulletish:
        units = bulletish
    else:
        units = [u.strip() for u in re.split(r"(?<=[.!?])\s+", text.strip()) if u.strip()]
    if not units:
        return None
    with_cite = sum(1 for u in units if _CITE_RE.search(u))
    return with_cite / max(1, len(units))


def _heuristic_quality_score(state: ResearchState) -> float | None:
    text = state.final_answer
    if not text:
        return None
    coverage = _citation_coverage(text) or 0.0
    has_structure = 1.0 if ("\n-" in text or "\n###" in text or "\n1." in text) else 0.0
    length_ok = 1.0 if len(text) >= 600 else 0.0
    score = 2.0 + 5.0 * coverage + 2.0 * has_structure + 1.0 * length_ok
    return max(0.0, min(10.0, score))


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return a placeholder metric object.

    TODO(student): Add quality scoring, estimated token cost, citation coverage, and error rate.
    """

    started = perf_counter()
    try:
        state = runner(query)
        success = True
        error = None
    except Exception as exc:  # noqa: BLE001
        state = ResearchState.model_validate({"request": {"query": query}})
        state.record_failure("benchmark", str(exc))
        success = False
        error = str(exc)
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.usage.cost_usd,
        quality_score=_heuristic_quality_score(state),
        citation_coverage=_citation_coverage(state.final_answer),
        success=success,
        error=error,
        input_tokens=state.usage.input_tokens,
        output_tokens=state.usage.output_tokens,
        notes="quality_score is heuristic; replace with peer review 0-10",
    )
    return state, metrics

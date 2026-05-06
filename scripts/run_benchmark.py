from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.utils.timer import elapsed_timer


def baseline_runner(query: str) -> ResearchState:
    settings = get_settings()
    state = ResearchState(request=ResearchQuery(query=query))
    llm = LLMClient(api_key=settings.openai_api_key, model=settings.openai_model, timeout_seconds=settings.timeout_seconds)
    system_prompt = (
        "You are a careful research assistant. Be concise, structured, and avoid fabrication. "
        "Include citations like [1] only if you have sources (you don't in baseline)."
    )
    with elapsed_timer() as elapsed:
        resp = llm.complete(system_prompt=system_prompt, user_prompt=query)
    state.set_final_answer(resp.content)
    state.add_usage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, cost_usd=resp.cost_usd)
    state.timing.wall_seconds = elapsed()
    return state


def multi_agent_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


def main() -> None:
    queries_path = Path("mock_data") / "benchmark_queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    all_metrics: list[BenchmarkMetrics] = []
    store = LocalArtifactStore()

    for item in queries:
        qid = item["id"]
        q = item["query"]

        _, m_base = run_benchmark(run_name=f"baseline:{qid}", query=q, runner=baseline_runner)
        m_base.query_id = qid
        all_metrics.append(m_base)

        _, m_multi = run_benchmark(run_name=f"multi-agent:{qid}", query=q, runner=multi_agent_runner)
        m_multi.query_id = qid
        all_metrics.append(m_multi)

    report = render_markdown_report(all_metrics)

    successes = [m for m in all_metrics if m.success]
    failure_rate = 1.0 - (len(successes) / max(1, len(all_metrics)))

    def _avg(xs: list[float]) -> float:
        return mean(xs) if xs else 0.0

    baseline_lat = _avg([m.latency_seconds for m in all_metrics if m.run_name.startswith("baseline:") and m.success])
    multi_lat = _avg([m.latency_seconds for m in all_metrics if m.run_name.startswith("multi-agent:") and m.success])

    baseline_cov = _avg(
        [m.citation_coverage for m in all_metrics if m.run_name.startswith("baseline:") and m.citation_coverage is not None]
    )
    multi_cov = _avg(
        [
            m.citation_coverage
            for m in all_metrics
            if m.run_name.startswith("multi-agent:") and m.citation_coverage is not None
        ]
    )

    interpretation = "\n".join(
        [
            "",
            "## Interpretation",
            "",
            f"- Average latency (baseline): **{baseline_lat:.2f}s**",
            f"- Average latency (multi-agent): **{multi_lat:.2f}s**",
            f"- Average citation coverage (baseline): **{baseline_cov:.2f}** (expected low/no citations)",
            f"- Average citation coverage (multi-agent): **{multi_cov:.2f}**",
            f"- Failure rate (all runs): **{failure_rate:.2%}**",
            "",
            "## Failure modes observed / expected",
            "",
            "- Search returns no sources (or low-quality sources) → weak grounding and low citation coverage.",
            "- Tool/API timeouts → workflow returns partial state; mitigate with retries/fallback and smaller prompts.",
            "- Overlong intermediate notes → higher latency/cost; mitigate with strict output constraints per agent.",
            "",
            "## Notes",
            "",
            "- `quality_score` in this report is **heuristic**; replace with peer review score (0–10) using the rubric.",
            "- Traces are exported under `reports/traces/` for each multi-agent run.",
            "",
        ]
    )

    content = report + interpretation
    store.write_text("benchmark_report.md", content)


if __name__ == "__main__":
    main()


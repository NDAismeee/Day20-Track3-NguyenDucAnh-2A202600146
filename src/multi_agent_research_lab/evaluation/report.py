"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown.

    TODO(student): Add richer analysis, examples, screenshots, and trace links.
    """

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Tokens (in/out) | Citation coverage | Success | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        tokens = ""
        if item.input_tokens is not None or item.output_tokens is not None:
            tokens = f"{item.input_tokens or 0}/{item.output_tokens or 0}"
        coverage = "" if item.citation_coverage is None else f"{item.citation_coverage:.2f}"
        success = "1" if item.success else "0"
        note = item.notes
        if item.error:
            note = (note + " | " if note else "") + f"error={item.error}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {tokens} | {coverage} | {success} | {note} |")
    return "\n".join(lines) + "\n"

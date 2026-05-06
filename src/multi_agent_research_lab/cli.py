"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.utils.timer import elapsed_timer

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    settings = get_settings()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = LLMClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.timeout_seconds,
    )
    system_prompt = (
        "You are a careful research assistant. Be concise, structured, and avoid fabrication. "
        "If the question is ambiguous, state assumptions."
    )
    with elapsed_timer() as elapsed:
        resp = llm.complete(system_prompt=system_prompt, user_prompt=state.request.query)

    state.set_final_answer(resp.content)
    state.add_usage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, cost_usd=resp.cost_usd)
    state.add_trace_event(
        "baseline_llm_complete",
        {
            "latency_seconds": elapsed(),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "model": settings.openai_model,
        },
    )

    metrics_line = (
        f"\n\n[dim]latency={elapsed():.2f}s"
        f" input_tokens={resp.input_tokens} output_tokens={resp.output_tokens}"
        f" model={settings.openai_model}[/dim]"
    )
    console.print(Panel.fit(f"{state.final_answer}{metrics_line}", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()

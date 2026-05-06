"""LangGraph workflow skeleton."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import export_trace_json, new_run_id, trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Create a LangGraph graph.

        TODO(student): Implement nodes, edges, conditional routing, and stop condition.
        Suggested nodes: supervisor, researcher, analyst, writer, optional critic.
        """

        graph: StateGraph[ResearchState] = StateGraph(ResearchState)
        run_id = new_run_id()

        def _usage_snapshot(s: ResearchState) -> tuple[int, int, float | None]:
            return (s.usage.input_tokens, s.usage.output_tokens, s.usage.cost_usd)

        def _usage_delta(before: tuple[int, int, float | None], after: tuple[int, int, float | None]) -> dict:
            in_before, out_before, cost_before = before
            in_after, out_after, cost_after = after
            delta: dict[str, object] = {
                "input_tokens": in_after - in_before,
                "output_tokens": out_after - out_before,
            }
            if cost_before is not None or cost_after is not None:
                delta["cost_usd"] = (cost_after or 0.0) - (cost_before or 0.0)
            return delta

        def supervisor_node(state: ResearchState) -> ResearchState:
            before = _usage_snapshot(state)
            attrs = {"iteration": state.iteration, "route_history_len": len(state.route_history)}
            try:
                with trace_span("supervisor", run_id=run_id, attributes=attrs) as span:
                    out = SupervisorAgent().run(state)
                    span.attributes["next_route"] = out.route_history[-1] if out.route_history else None
                    span.attributes["usage_delta"] = _usage_delta(before, _usage_snapshot(out))
                    out.add_trace_event("span", span.model_dump())
                    return out
            except Exception as exc:  # noqa: BLE001
                state.record_failure("supervisor", str(exc))
                state.add_trace_event("node_error", {"node": "supervisor", "error": str(exc)})
                state.record_route("done")
                return state

        def researcher_node(state: ResearchState) -> ResearchState:
            before = _usage_snapshot(state)
            attrs = {"query": state.request.query, "max_sources": state.request.max_sources}
            try:
                with trace_span("researcher", run_id=run_id, attributes=attrs) as span:
                    out = ResearcherAgent().run(state)
                    span.attributes["num_sources"] = len(out.sources)
                    span.attributes["research_notes_len"] = len(out.research_notes or "")
                    span.attributes["usage_delta"] = _usage_delta(before, _usage_snapshot(out))
                    out.add_trace_event("span", span.model_dump())
                    return out
            except Exception as exc:  # noqa: BLE001
                state.record_failure("researcher", str(exc))
                state.add_trace_event("node_error", {"node": "researcher", "error": str(exc)})
                return state

        def analyst_node(state: ResearchState) -> ResearchState:
            before = _usage_snapshot(state)
            attrs = {"research_notes_len": len(state.research_notes or "")}
            try:
                with trace_span("analyst", run_id=run_id, attributes=attrs) as span:
                    out = AnalystAgent().run(state)
                    span.attributes["analysis_notes_len"] = len(out.analysis_notes or "")
                    span.attributes["usage_delta"] = _usage_delta(before, _usage_snapshot(out))
                    out.add_trace_event("span", span.model_dump())
                    return out
            except Exception as exc:  # noqa: BLE001
                state.record_failure("analyst", str(exc))
                state.add_trace_event("node_error", {"node": "analyst", "error": str(exc)})
                return state

        def writer_node(state: ResearchState) -> ResearchState:
            before = _usage_snapshot(state)
            attrs = {
                "sources": len(state.sources),
                "research_notes_len": len(state.research_notes or ""),
                "analysis_notes_len": len(state.analysis_notes or ""),
            }
            try:
                with trace_span("writer", run_id=run_id, attributes=attrs) as span:
                    out = WriterAgent().run(state)
                    span.attributes["final_answer_len"] = len(out.final_answer or "")
                    span.attributes["usage_delta"] = _usage_delta(before, _usage_snapshot(out))
                    out.add_trace_event("span", span.model_dump())
                    return out
            except Exception as exc:  # noqa: BLE001
                state.record_failure("writer", str(exc))
                state.add_trace_event("node_error", {"node": "writer", "error": str(exc)})
                return state

        def route_from_supervisor(state: ResearchState) -> str:
            if state.final_answer:
                return "done"
            if not state.route_history:
                return "supervisor"
            return state.route_history[-1]

        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        compiled = graph.compile()
        compiled.run_id = run_id  # type: ignore[attr-defined]
        return compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        TODO(student): Compile graph, invoke it, and convert result back to ResearchState.
        """

        settings = get_settings()
        app = self.build()
        run_id = getattr(app, "run_id", new_run_id())
        state.add_trace_event("run_started", {"run_id": run_id, "query": state.request.query})

        recursion_limit = max(10, settings.max_iterations * 4)

        def _invoke() -> ResearchState:
            result = app.invoke(state, config={"recursion_limit": recursion_limit})
            return result if isinstance(result, ResearchState) else ResearchState.model_validate(result)

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_invoke)
            try:
                result = fut.result(timeout=settings.timeout_seconds)
                trace_path = export_trace_json(run_id, result.trace)
                result.add_trace_event("trace_exported", {"path": trace_path})
                return result
            except FutureTimeoutError:
                state.record_failure("workflow", "timeout")
                state.add_trace_event("workflow_timeout", {"timeout_seconds": settings.timeout_seconds})
                trace_path = export_trace_json(run_id, state.trace)
                state.add_trace_event("trace_exported", {"path": trace_path})
                return state

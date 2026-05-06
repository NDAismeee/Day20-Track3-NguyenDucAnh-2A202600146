"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        TODO(student): Implement routing policy. Suggested steps:
        - Inspect request, current notes, and missing fields.
        - Choose one of: researcher, analyst, writer, done.
        - Enforce max iterations and failure fallback.
        """

        settings = get_settings()

        if state.final_answer:
            state.add_trace_event("supervisor_decision", {"route": "done", "reason": "final_answer_present"})
            state.record_route("done")
            return state

        if state.iteration >= settings.max_iterations:
            state.record_failure("supervisor", "max_iterations_exceeded")
            state.add_trace_event(
                "supervisor_decision",
                {"route": "done", "reason": "max_iterations_exceeded", "max_iterations": settings.max_iterations},
            )
            state.record_route("done")
            return state

        if not state.research_notes or not state.sources:
            candidate = "researcher"
            reason = "missing_research"
        elif not state.analysis_notes:
            candidate = "analyst"
            reason = "missing_analysis"
        else:
            candidate = "writer"
            reason = "ready_to_write"

        failures = state.failure_counts.get(candidate, 0)
        if failures >= 2:
            if candidate != "writer":
                state.add_trace_event(
                    "supervisor_fallback",
                    {"from": candidate, "to": "writer", "reason": "too_many_failures", "failures": failures},
                )
                candidate = "writer"
                reason = "fallback_to_writer"
            else:
                state.record_failure("supervisor", "writer_failed_too_many_times")
                state.add_trace_event(
                    "supervisor_decision",
                    {"route": "done", "reason": "writer_failed_too_many_times", "failures": failures},
                )
                state.record_route("done")
                return state

        state.add_trace_event(
            "supervisor_decision",
            {"route": candidate, "reason": reason, "iteration": state.iteration, "failures": failures},
        )
        state.record_route(candidate)
        return state

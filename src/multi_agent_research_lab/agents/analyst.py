"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.utils.timer import elapsed_timer


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        TODO(student): Extract key claims, compare viewpoints, and flag weak evidence.
        """

        if not state.research_notes:
            raise AgentExecutionError("Missing research_notes")

        settings = get_settings()
        llm = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.timeout_seconds,
        )

        system_prompt = (
            "You are an analyst. Turn research notes into structured analysis. "
            "Keep citations where applicable using the same [n] markers already present."
        )
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            "Write analysis_notes with these sections:\n"
            "1) Key claims (5-8 bullets, keep citations)\n"
            "2) Pros / benefits (3-6 bullets)\n"
            "3) Cons / risks (3-6 bullets)\n"
            "4) Reasoning / synthesis (1 short paragraph)\n"
            "5) Gaps / what to verify next (3-6 bullets)\n"
        )

        with elapsed_timer() as t_llm:
            resp = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.record_stage_timing("analyst_llm", t_llm())
        state.add_usage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, cost_usd=resp.cost_usd)

        notes = (resp.content or "").strip()
        if not notes:
            raise ValidationError("analysis_notes must be non-empty")
        state.analysis_notes = notes
        state.add_trace_event(
            "analyst_notes",
            {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "model": settings.openai_model},
        )
        return state

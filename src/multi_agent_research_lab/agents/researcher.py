"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.utils.timer import elapsed_timer


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        TODO(student): Implement search, source filtering, citation capture, and notes.
        """

        settings = get_settings()
        search = SearchClient()
        llm = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.timeout_seconds,
        )

        with elapsed_timer() as t_search:
            sources = search.search(state.request.query, max_results=state.request.max_sources)
        state.sources = sources
        state.record_stage_timing("researcher_search", t_search())
        state.add_trace_event(
            "researcher_search",
            {"num_sources": len(sources), "max_sources": state.request.max_sources},
        )

        if not sources:
            raise AgentExecutionError("No sources found")

        sources_block = "\n".join(
            [
                f"[{i+1}] {s.title} | {s.url or 'NO_URL'}\n{s.snippet}"
                for i, s in enumerate(sources)
            ]
        )
        system_prompt = (
            "You are a researcher. Use ONLY the provided sources. "
            "Write concise research notes. Every key claim must cite sources like [1][2]. "
            "If evidence is weak, explicitly say so."
        )
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Produce research_notes:\n"
            "- 8-12 bullets\n"
            "- each bullet ends with citations like [1]\n"
            "- include any important caveats\n"
        )

        with elapsed_timer() as t_llm:
            resp = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.record_stage_timing("researcher_llm", t_llm())
        state.add_usage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, cost_usd=resp.cost_usd)

        notes = (resp.content or "").strip()
        if not notes:
            raise ValidationError("research_notes must be non-empty")
        state.research_notes = notes
        state.add_trace_event(
            "researcher_notes",
            {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "model": settings.openai_model},
        )
        return state

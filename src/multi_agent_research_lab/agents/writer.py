"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.utils.timer import elapsed_timer


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        TODO(student): Synthesize a clear response with citations or source references.
        """

        if not state.research_notes:
            raise AgentExecutionError("Missing research_notes")

        settings = get_settings()
        llm = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.timeout_seconds,
        )

        sources_index = "\n".join(
            [f"[{i+1}] {s.title} | {s.url or 'NO_URL'}" for i, s in enumerate(state.sources)]
        )

        system_prompt = (
            "You are a technical writer. Produce a clear final answer grounded in the notes. "
            "Keep citations like [1] inline for claims that came from sources. "
            "Do not invent sources; only use the provided source indices."
        )
        analysis = state.analysis_notes or "(no analysis provided)"
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Source index:\n{sources_index}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis notes:\n{analysis}\n\n"
            "Write the final answer with:\n"
            "- short intro\n"
            "- 5-8 bullet points or a structured outline\n"
            "- a brief 'What to verify next' section\n"
            "- citations [n] on key claims\n"
        )

        with elapsed_timer() as t_llm:
            resp = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.record_stage_timing("writer_llm", t_llm())
        state.add_usage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, cost_usd=resp.cost_usd)

        state.set_final_answer(resp.content)
        state.add_trace_event(
            "writer_final",
            {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "model": settings.openai_model},
        )
        return state

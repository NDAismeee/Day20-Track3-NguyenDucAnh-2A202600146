"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic import field_validator

from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import (
    AgentResult,
    ChatMessage,
    ResearchQuery,
    SourceDocument,
    TimingMetrics,
    UsageMetrics,
)


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    messages: list[ChatMessage] = Field(default_factory=list)

    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)
    failure_counts: dict[str, int] = Field(default_factory=dict)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None

    agent_results: list[AgentResult] = Field(default_factory=list)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    timing: TimingMetrics = Field(default_factory=TimingMetrics)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("research_notes", "analysis_notes", "final_answer")
    @classmethod
    def _strip_optional_text(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    def record_route(self, route: str) -> None:
        self.route_history.append(route)
        self.iteration += 1

    def record_failure(self, route: str, message: str) -> None:
        self.failure_counts[route] = self.failure_counts.get(route, 0) + 1
        self.errors.append(f"{route}: {message}")

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def add_usage(self, *, input_tokens: int | None, output_tokens: int | None, cost_usd: float | None = None) -> None:
        if input_tokens is not None:
            self.usage.input_tokens += input_tokens
        if output_tokens is not None:
            self.usage.output_tokens += output_tokens
        if cost_usd is not None:
            self.usage.cost_usd = (self.usage.cost_usd or 0.0) + cost_usd

    def record_stage_timing(self, stage: str, seconds: float) -> None:
        if seconds < 0:
            raise ValidationError("Timing seconds must be non-negative")
        self.timing.by_stage_seconds[stage] = seconds

    def set_final_answer(self, content: str) -> None:
        cleaned = content.strip()
        if not cleaned:
            raise ValidationError("final_answer must be non-empty")
        self.final_answer = cleaned

"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.services.storage import LocalArtifactStore


class TraceSpan(BaseModel):
    run_id: str
    span_id: str
    name: str
    started_at: str
    duration_seconds: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
    error: str | None = None


def new_run_id() -> str:
    return uuid4().hex


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_trace_json(run_id: str, trace: list[dict[str, Any]]) -> str:
    store = LocalArtifactStore()
    path = store.write_text(f"traces/{run_id}.json", content=__import__("json").dumps(trace, indent=2))
    return str(path)


@contextmanager
def trace_span(
    name: str,
    *,
    run_id: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[TraceSpan]:
    """Minimal span context used by the skeleton.

    TODO(student): Replace or augment with LangSmith/Langfuse provider spans.
    """

    started = perf_counter()
    span = TraceSpan(
        run_id=run_id,
        span_id=uuid4().hex,
        name=name,
        started_at=utc_now_iso(),
        attributes=attributes or {},
        duration_seconds=None,
    )
    try:
        yield span
    except Exception as exc:  # noqa: BLE001
        span.status = "error"
        span.error = str(exc)
        raise
    finally:
        span.duration_seconds = perf_counter() - started

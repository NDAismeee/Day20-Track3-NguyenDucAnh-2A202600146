"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self, *, api_key: str | None, model: str, timeout_seconds: int = 60) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        TODO(student): Connect OpenAI, Azure OpenAI, or another provider.
        Keep retry, timeout, and token logging here rather than inside agents.
        """

        if not self._api_key:
            raise StudentTodoError("Missing OPENAI_API_KEY in .env")

        try:
            from openai import OpenAI
        except Exception as exc:  # noqa: BLE001
            raise StudentTodoError(
                "OpenAI SDK not installed. Install optional deps: pip install -e '.[llm]'"
            ) from exc

        try:
            client = OpenAI(api_key=self._api_key, timeout=self._timeout_seconds)
            resp = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc

        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return LLMResponse(content=content, input_tokens=input_tokens, output_tokens=output_tokens)

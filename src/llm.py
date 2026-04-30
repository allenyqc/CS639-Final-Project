from __future__ import annotations

import os

# Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic in .env to force a provider.
# If unset, auto-detect: use Anthropic if ANTHROPIC_API_KEY is present, else OpenAI.
_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
if not _PROVIDER:
    _PROVIDER = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if _PROVIDER == "openai":
        import openai
        _client = openai.OpenAI()
    else:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def call_llm(system: str, prompt: str) -> str:
    client = _get_client()
    if _PROVIDER == "openai":
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

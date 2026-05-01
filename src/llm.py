from __future__ import annotations

import os
import time

# Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic in .env to force a provider.
# If unset, auto-detect: use Anthropic if ANTHROPIC_API_KEY is present, else OpenAI.
_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
if not _PROVIDER:
    _PROVIDER = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"

_MODEL = os.getenv("LLM_MODEL", "")

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2  # seconds, exponential backoff

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


def _default_model() -> str:
    if _MODEL:
        return _MODEL
    return "gpt-4o-2024-11-20" if _PROVIDER == "openai" else "claude-sonnet-4-6"


def call_llm(system: str, prompt: str) -> str:
    client = _get_client()
    model = _default_model()

    for attempt in range(_MAX_RETRIES):
        try:
            if _PROVIDER == "openai":
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            else:
                response = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.2,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"LLM call failed after {_MAX_RETRIES} attempts: {e}"
                ) from e
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  [retry] attempt {attempt + 1} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError("Unreachable")

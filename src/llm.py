from __future__ import annotations

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
if not _PROVIDER:
    _PROVIDER = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"

_MODEL = os.getenv("LLM_MODEL", "")

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2

_OPUS_47_MODELS = {"claude-opus-4-7", "claude-opus-4-7-20250430"}
_OPUS_47_MAX_RETRIES = 3
_OPUS_47_RETRY_BASE_DELAY = 4
_OPUS_47_MAX_TOKENS = 32768
_OPUS_47_TIMEOUT = 600.0

_client = None
_opus_client = None


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


def _get_opus_client():
    """Separate client for Opus 4.7 with extended timeout for xhigh thinking."""
    global _opus_client
    if _opus_client is not None:
        return _opus_client
    import anthropic
    _opus_client = anthropic.Anthropic(
        timeout=_OPUS_47_TIMEOUT,
    )
    return _opus_client


def _default_model() -> str:
    if _MODEL:
        return _MODEL
    return "gpt-4o-2024-11-20" if _PROVIDER == "openai" else "claude-sonnet-4-6"


def _is_opus_47(model: str) -> bool:
    return model in _OPUS_47_MODELS


def _extract_text(response_content: list) -> str:
    """Extract text from Anthropic response content, skipping thinking blocks."""
    for block in response_content:
        if block.type == "text":
            return block.text
    return response_content[-1].text


def _log_usage(response, model: str) -> None:
    """Print token usage for cost tracking."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    input_tok = getattr(usage, "input_tokens", 0)
    output_tok = getattr(usage, "output_tokens", 0)

    parts = [f"in={input_tok}", f"out={output_tok}"]

    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    if cache_read:
        parts.append(f"cache_hit={cache_read}")

    print(f"    [tokens] {model}: {', '.join(parts)}")


def _is_rate_limit(exc: Exception) -> bool:
    """Check if an exception is a rate-limit (429) or overloaded (529) error."""
    exc_str = str(type(exc).__name__).lower()
    if "ratelimit" in exc_str or "overloaded" in exc_str:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (429, 529):
        return True
    return False


def call_llm(system: str, prompt: str) -> str:
    model = _default_model()
    is_opus = _is_opus_47(model)

    max_retries = _OPUS_47_MAX_RETRIES if is_opus else _MAX_RETRIES
    base_delay = _OPUS_47_RETRY_BASE_DELAY if is_opus else _RETRY_BASE_DELAY

    for attempt in range(max_retries):
        try:
            if _PROVIDER == "openai":
                client = _get_client()
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
                if is_opus:
                    client = _get_opus_client()
                    response = client.messages.create(
                        model=model,
                        max_tokens=_OPUS_47_MAX_TOKENS,
                        thinking={"type": "adaptive"},
                        output_config={"effort": "xhigh"},
                        system=system,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    _log_usage(response, model)

                    if response.stop_reason == "max_tokens":
                        print(f"    [warn] response truncated at {_OPUS_47_MAX_TOKENS} tokens")

                    return _extract_text(response.content)
                else:
                    client = _get_client()
                    response = client.messages.create(
                        model=model,
                        max_tokens=5250,
                        temperature=0.2,
                        system=system,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"LLM call failed after {max_retries} attempts: {e}"
                ) from e
            delay = base_delay * (2 ** attempt)
            if _is_rate_limit(e):
                delay = max(delay, 30)
                print(f"  [rate-limit] attempt {attempt + 1} hit rate limit, waiting {delay}s...")
            else:
                print(f"  [retry] attempt {attempt + 1} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError("Unreachable")

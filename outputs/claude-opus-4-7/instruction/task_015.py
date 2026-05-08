import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def fetch_with_retry(
    url: str,
    headers: Optional[dict] = None,
    max_attempts: int = 3,
    timeout: float = 10.0,
) -> Optional[str]:
    """Fetch a URL with retries and exponential backoff.

    Args:
        url: The URL to fetch.
        headers: Optional HTTP headers to include in the request.
        max_attempts: Maximum number of attempts before giving up.
        timeout: Per-request timeout in seconds.

    Returns:
        The response text on success, or None if all attempts fail.
    """
    request_headers = dict(headers) if headers else {}

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.Session() as session:
                response = session.get(
                    url,
                    headers=request_headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.text
        except (
            requests.exceptions.RequestException,
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            logger.warning(
                "Attempt %d/%d to fetch %s failed: %s",
                attempt,
                max_attempts,
                url,
                exc,
            )
            if attempt < max_attempts:
                backoff = 2 ** (attempt - 1)
                time.sleep(backoff)

    logger.error("All %d attempts to fetch %s failed.", max_attempts, url)
    return None
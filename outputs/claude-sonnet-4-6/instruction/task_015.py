import logging
import time
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_with_retry(url: str, headers: list | None = None) -> str | None:
    """
    Fetch a URL with up to 3 retries and exponential backoff.

    Args:
        url: The URL to fetch.
        headers: Optional list of (key, value) tuples to include as request headers.

    Returns:
        The response text on success, or None if all attempts fail.
    """
    if headers is None:
        headers = []

    headers_dict = dict(headers)

    max_attempts = 3
    backoff_seconds = 1

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.Session() as session:
                response = session.get(url, headers=headers_dict, timeout=10)
                response.raise_for_status()
                return response.text

        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "Attempt %d/%d failed with HTTP error: %s",
                attempt,
                max_attempts,
                exc,
            )
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Attempt %d/%d failed with connection error: %s",
                attempt,
                max_attempts,
                exc,
            )
        except requests.exceptions.Timeout as exc:
            logger.warning(
                "Attempt %d/%d failed with timeout: %s",
                attempt,
                max_attempts,
                exc,
            )
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Attempt %d/%d failed with request error: %s",
                attempt,
                max_attempts,
                exc,
            )

        if attempt < max_attempts:
            logger.info("Retrying in %d second(s)...", backoff_seconds)
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

    logger.error("All %d attempts failed for URL: %s", max_attempts, url)
    return None
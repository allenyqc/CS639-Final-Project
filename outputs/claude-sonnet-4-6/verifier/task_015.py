import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_with_retry(url: str, headers: list[dict] | None = None) -> str | None:
    """
    Fetch a URL with retry logic and exponential backoff.
    
    Args:
        url: The URL to fetch.
        headers: Optional list of header dictionaries to include in the request.
    
    Returns:
        The response text on success, or None if all attempts fail.
    """
    max_retries = 3
    backoff_times = [1, 2, 4]
    
    # Merge headers if provided
    merged_headers = {}
    if headers:
        for header_dict in headers:
            if isinstance(header_dict, dict):
                merged_headers.update(header_dict)
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt} of {max_retries}: Fetching URL: {url}")
            response = requests.get(url, headers=merged_headers if merged_headers else None, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully fetched URL on attempt {attempt}")
            return response.text
        
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Attempt {attempt} failed with HTTP error: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Attempt {attempt} failed with connection error: {e}")
        except requests.exceptions.Timeout as e:
            logger.warning(f"Attempt {attempt} failed with timeout error: {e}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt} failed with error: {e}")
        
        if attempt < max_retries:
            wait_time = backoff_times[attempt - 1]
            logger.info(f"Waiting {wait_time} second(s) before next attempt...")
            time.sleep(wait_time)
    
    logger.error(f"All {max_retries} attempts failed for URL: {url}")
    return None


if __name__ == "__main__":
    # Example usage
    result = fetch_with_retry(
        "https://httpbin.org/get",
        headers=[{"User-Agent": "MyApp/1.0"}, {"Accept": "application/json"}]
    )
    if result:
        print("Success! Response length:", len(result))
    else:
        print("Failed to fetch URL")
    
    # Test with invalid URL to see retry behavior
    print("\nTesting with invalid URL:")
    result = fetch_with_retry("https://this-url-does-not-exist-12345.com")
    print("Result:", result)
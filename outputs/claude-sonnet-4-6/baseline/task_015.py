import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_with_retry(url: str, headers: list = None) -> str | None:
    """
    Fetch a URL with retry logic and exponential backoff.
    
    Args:
        url: The URL to fetch
        headers: Optional list of headers (as a dict or list of dicts)
    
    Returns:
        Response text on success, None if all attempts fail
    """
    max_retries = 3
    backoff_times = [1, 2, 4]
    
    # Handle headers - convert list to dict if necessary
    request_headers = {}
    if headers is not None:
        if isinstance(headers, dict):
            request_headers = headers
        elif isinstance(headers, list):
            for header in headers:
                if isinstance(header, dict):
                    request_headers.update(header)
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt} of {max_retries}: Fetching URL: {url}")
            response = requests.get(url, headers=request_headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully fetched URL on attempt {attempt}")
            return response.text
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {e.response.status_code} - {str(e)}"
            logger.warning(f"Attempt {attempt} failed - {error_msg}")
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.warning(f"Attempt {attempt} failed - {error_msg}")
        
        except requests.exceptions.Timeout as e:
            error_msg = f"Request timed out: {str(e)}"
            logger.warning(f"Attempt {attempt} failed - {error_msg}")
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            logger.warning(f"Attempt {attempt} failed - {error_msg}")
        
        # If not the last attempt, wait before retrying
        if attempt < max_retries:
            wait_time = backoff_times[attempt - 1]
            logger.info(f"Waiting {wait_time} second(s) before retry...")
            time.sleep(wait_time)
    
    logger.error(f"All {max_retries} attempts failed for URL: {url}")
    return None


if __name__ == "__main__":
    # Example usage
    result = fetch_with_retry("https://httpbin.org/get")
    if result:
        print("Success! Response length:", len(result))
    else:
        print("Failed to fetch URL")
    
    # Example with headers
    headers = {"User-Agent": "MyBot/1.0", "Accept": "application/json"}
    result = fetch_with_retry("https://httpbin.org/headers", headers=headers)
    if result:
        print("\nWith headers - Success! Response length:", len(result))
    
    # Example with invalid URL to demonstrate retry
    result = fetch_with_retry("https://this-url-does-not-exist-12345.com")
    if result is None:
        print("\nExpected failure for invalid URL - returned None")
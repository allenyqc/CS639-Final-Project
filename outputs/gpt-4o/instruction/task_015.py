import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def fetch_with_retry(url, headers=None):
    if headers is None:
        headers = {}

    max_attempts = 3
    backoff_times = [1, 2, 4]

    for attempt in range(max_attempts):
        try:
            with requests.get(url, headers=headers) as response:
                response.raise_for_status()
                return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(backoff_times[attempt])
    
    return None
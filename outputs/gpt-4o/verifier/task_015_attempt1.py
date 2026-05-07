import requests
import time
import logging

logging.basicConfig(level=logging.INFO)

def fetch_with_retry(url, headers=None):
    max_attempts = 3
    backoff_times = [1, 2, 4]
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.info(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(backoff_times[attempt - 1])
    
    return None
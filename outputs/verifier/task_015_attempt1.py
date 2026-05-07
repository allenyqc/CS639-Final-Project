import requests
import time
import logging

logging.basicConfig(level=logging.INFO)

def fetch_with_retry(url, headers=None):
    attempts = 3
    backoff = 1

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt} failed: {e}")
            if attempt < attempts:
                time.sleep(backoff)
                backoff *= 2
            else:
                return None
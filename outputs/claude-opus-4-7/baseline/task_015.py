import logging
import time
import requests


def fetch_with_retry(url, headers=None):
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))
    return None
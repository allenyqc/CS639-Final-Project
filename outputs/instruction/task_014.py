import json
import os

def load_json_config(file_path):
    """
    Loads a JSON configuration file and returns the parsed dictionary.
    If the file does not exist or contains invalid JSON, returns an empty dictionary.

    :param file_path: Path to the JSON configuration file.
    :return: Parsed dictionary or an empty dictionary on error.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")

    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError:
        # File exists but contains invalid JSON
        return {}
    except OSError:
        # Handle file-related errors (e.g., permission issues)
        return {}
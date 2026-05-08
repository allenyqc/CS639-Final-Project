import json
import os


def load_json_config(file_path):
    """
    Load a JSON configuration file and return the parsed dictionary.

    Args:
        file_path (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed dictionary from the JSON file, or an empty dictionary
              if the file does not exist or contains invalid JSON.
    """
    if not os.path.isfile(file_path):
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    return data
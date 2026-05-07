import json
import os


def load_json_config(file_path: str) -> dict:
    """
    Load a JSON configuration file from the given file path.

    Returns the parsed dictionary on success, or an empty dictionary
    if the file does not exist or contains invalid JSON.
    """
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except (json.JSONDecodeError, OSError):
        return {}
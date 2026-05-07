import json
import os


def load_config(file_path: str) -> dict:
    """
    Load a JSON configuration file and return the parsed dictionary.
    
    Args:
        file_path: Path to the JSON configuration file.
    
    Returns:
        Parsed dictionary from the JSON file, or an empty dictionary
        if the file does not exist or contains invalid JSON.
    """
    if not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}
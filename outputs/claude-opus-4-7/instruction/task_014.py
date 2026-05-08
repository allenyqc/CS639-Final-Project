import json
from pathlib import Path
from typing import Any, Union


def load_json_config(file_path: Union[str, Path]) -> dict[str, Any]:
    """Load a JSON configuration file and return the parsed dictionary.

    Args:
        file_path: Path to the JSON configuration file.

    Returns:
        Parsed dictionary from the JSON file, or an empty dictionary if the
        file does not exist or contains invalid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as config_file:
            parsed = json.load(config_file)
    except (FileNotFoundError, IsADirectoryError, PermissionError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(parsed, dict):
        return {}

    return parsed
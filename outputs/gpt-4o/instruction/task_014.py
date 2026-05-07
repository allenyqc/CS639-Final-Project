import json
import os

def load_json_config(file_path):
    if not isinstance(file_path, str):
        return {}
    
    if not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
            if isinstance(config_data, dict):
                return config_data
            else:
                return {}
    except json.JSONDecodeError:
        return {}
    except OSError:
        return {}

# Example usage:
# config = load_json_config('config.json')
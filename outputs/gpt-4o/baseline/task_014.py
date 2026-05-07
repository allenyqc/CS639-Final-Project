import json
import os

def load_json_config(file_path):
    if not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, IOError):
        return {}
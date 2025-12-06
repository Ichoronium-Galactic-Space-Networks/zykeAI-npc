import os
import json

def save_config(config, path):
    """
    Save a configuration dictionary to a JSON file.
    
    Args:
        config (dict): The configuration dictionary.
        path (str): The file path to save the configuration to.
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(path):
    """
    Load a configuration dictionary from a JSON file.
    
    Args:
        path (str): The file path to load the configuration from.
    
    Returns:
        dict: The loaded configuration dictionary.
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config

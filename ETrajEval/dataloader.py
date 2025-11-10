# dataloader.py

import json
from typing import List, Dict, Any

def load_structured_prompts(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads structured prompts from a JSON file. Each object in the JSON list should be a dictionary containing
    the necessary prompts to start a conversation.
    """
    print(f"[*] Loading structured prompts from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        print(f"[+] Successfully loaded {len(prompts)} prompt scenarios.")
        return prompts
    except FileNotFoundError:
        print(f"[!] Error: Prompt file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"[!] Error: Could not decode JSON from {file_path}. Please check its format.")
        return []
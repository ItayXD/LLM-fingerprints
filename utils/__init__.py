"""Utils package for model compression and bit limiting research."""

import os
from pathlib import Path

# Import key functions from llm_api for easy access
from .llm_api import (get_anthropic_key, get_openai_key, check_api_keys,
                      anthropic_completion, openai_completion,
                      anthropic_messages, openai_messages, anthropic_stream,
                      openai_stream, get_available_models)

# Define the SECRETS directory path
SECRETS_DIR = Path(__file__).parent.parent / "SECRETS"

# Ensure SECRETS directory exists
SECRETS_DIR.mkdir(exist_ok=True)

# Export key constants and functions
__all__ = [
    "SECRETS_DIR", "get_anthropic_key", "get_openai_key", "check_api_keys",
    "anthropic_completion", "openai_completion", "anthropic_messages",
    "openai_messages", "anthropic_stream", "openai_stream",
    "get_available_models"
]

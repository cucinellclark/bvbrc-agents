"""
LLM Client for internal model calls
"""
import requests
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Make the shared config importable
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "config")
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)

from llm_config import get_excluded_params


class LLMClient:
    """Client for making calls to a local LLM endpoint."""
    
    def __init__(self, endpoint: str, model: str, api_key: str = "EMPTY", 
                 temperature: float = 0.7, max_tokens: int = 2000, timeout: int = 60):
        """
        Initialize LLM client.
        
        Args:
            endpoint: Full URL to the chat completions endpoint
            model: Model identifier
            api_key: API key (use "EMPTY" for local models)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> str:
        """
        Make a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            The assistant's response text
            
        Raises:
            requests.RequestException: If the request fails
            ValueError: If the response is invalid
        """
        excluded = get_excluded_params(self.model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if "temperature" not in excluded:
            payload["temperature"] = temperature if temperature is not None else self.temperature
        if "max_tokens" not in excluded:
            payload["max_tokens"] = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the assistant's message
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
        except requests.RequestException as e:
            print(f"LLM request failed: {e}", file=sys.stderr)
            raise
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Failed to parse LLM response: {e}", file=sys.stderr)
            raise ValueError(f"Invalid LLM response: {e}")


def create_llm_client_from_config(config: Dict) -> LLMClient:
    """
    Create an LLM client from configuration dictionary.
    
    Args:
        config: Configuration dict with 'llm' section
        
    Returns:
        Configured LLMClient instance
    """
    llm_config = config.get("llm", {})
    
    return LLMClient(
        endpoint=llm_config.get("endpoint", "http://localhost:8000/v1/chat/completions"),
        model=llm_config.get("model", "default-model"),
        api_key=llm_config.get("api_key", "EMPTY"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 2000),
        timeout=llm_config.get("timeout", 60)
    )

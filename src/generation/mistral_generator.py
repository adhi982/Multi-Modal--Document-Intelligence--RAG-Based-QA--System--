"""Mistral API integration for generation."""

import os
from typing import Optional
import requests
import time
from dotenv import load_dotenv

from ..utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


class MistralGenerator:
    """Mistral API based generator."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-medium-latest",
        max_tokens: int = 512,
        temperature: float = 0.3
    ):
        """
        Initialize Mistral generator.
        
        Args:
            api_key: Mistral API key (from env if not provided)
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.logger = logger
        
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY in .env file")
        
        self.logger.info(f"Initialized Mistral API generator with model: {model}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_beams: int = 1,
        retry_attempts: int = 3
    ) -> str:
        """
        Generate response using Mistral API.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override max tokens
            temperature: Override temperature
            num_beams: Ignored (not supported by API)
            retry_attempts: Number of retry attempts on failure
            
        Returns:
            Generated text
        """
        max_tokens = max_new_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temp
        }
        
        for attempt in range(retry_attempts):
            try:
                self.logger.info(f"Calling Mistral API (attempt {attempt + 1}/{retry_attempts})...")
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                response.raise_for_status()
                
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                
                self.logger.info(f"Generated {len(generated_text)} characters")
                return generated_text
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Mistral API request failed (attempt {attempt + 1}): {e}")
                
                if attempt < retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("All Mistral API attempts failed")
                    raise Exception(f"Mistral API failed after {retry_attempts} attempts: {e}")
            
            except Exception as e:
                self.logger.error(f"Unexpected error calling Mistral API: {e}")
                raise

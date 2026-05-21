"""
LLM Client Module for the RAG Q&A System.

Provides a unified interface for both Groq (cloud API) and Ollama (local)
LLM providers. Supports switching between providers via environment
variables for maximum flexibility.
"""

import time
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from src.config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL,
    OLLAMA_BASE_URL, RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE,
    BASELINE_SYSTEM_PROMPT, BASELINE_USER_PROMPT_TEMPLATE,
    logger
)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction prompt
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with keys: 'response', 'model', 'latency_seconds',
            'input_tokens', 'output_tokens'
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """Return provider information."""
        pass


class GroqClient(BaseLLMClient):
    """
    Groq API client for cloud-based LLM inference.
    
    Uses Groq's ultra-fast inference API. Requires a free API key
    from https://console.groq.com/keys
    
    Supported models:
    - llama-3.1-8b-instant
    - mixtral-8x7b-32768
    - gemma2-9b-it
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the Groq client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY from env)
            model: Model name (defaults to GROQ_MODEL from env)
        """
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            raise ValueError(
                "Groq API key not set. Get a free key at https://console.groq.com/keys "
                "and set GROQ_API_KEY in your .env file."
            )
        
        from groq import Groq
        self.client = Groq(api_key=self.api_key)
        logger.info(f"Groq client initialized with model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict:
        """Generate response using Groq API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Avoid Groq free tier rate limits (30 RPM)
        time.sleep(2.5)
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            latency = time.time() - start_time
            
            return {
                "response": response.choices[0].message.content,
                "model": self.model,
                "provider": "groq",
                "latency_seconds": round(latency, 3),
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            }
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {
                "response": f"Error: {str(e)}",
                "model": self.model,
                "provider": "groq",
                "latency_seconds": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }
    
    def get_info(self) -> Dict:
        return {
            "provider": "groq",
            "model": self.model,
            "api_key_set": bool(self.api_key),
        }


class OllamaClient(BaseLLMClient):
    """
    Ollama client for local LLM inference.
    
    Runs models locally using Ollama. Requires Ollama to be installed
    and running, with the desired model pulled.
    
    Setup:
        1. Install Ollama: https://ollama.ai
        2. Pull model: ollama pull llama3.1:8b
        3. Ollama runs automatically on http://localhost:11434
    """
    
    def __init__(self, model: str = None, base_url: str = None):
        """
        Initialize the Ollama client.
        
        Args:
            model: Model name (defaults to OLLAMA_MODEL from env)
            base_url: Ollama server URL (defaults to OLLAMA_BASE_URL from env)
        """
        self.model = model or OLLAMA_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        
        import ollama
        self.client = ollama.Client(host=self.base_url)
        
        # Verify connection
        try:
            self.client.list()
            logger.info(f"Ollama client initialized with model: {self.model}")
        except Exception as e:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}: {e}. "
                f"Make sure Ollama is running: 'ollama serve'"
            )
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict:
        """Generate response using local Ollama model."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            
            latency = time.time() - start_time
            
            return {
                "response": response["message"]["content"],
                "model": self.model,
                "provider": "ollama",
                "latency_seconds": round(latency, 3),
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
            }
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return {
                "response": f"Error: {str(e)}",
                "model": self.model,
                "provider": "ollama",
                "latency_seconds": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }
    
    def get_info(self) -> Dict:
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
        }


def get_llm_client(
    provider: str = None,
    model: str = None,
    api_key: str = None,
) -> BaseLLMClient:
    """
    Factory function to create the appropriate LLM client.
    
    Creates either a GroqClient or OllamaClient based on the provider
    setting. Defaults to environment configuration.
    
    Args:
        provider: "groq" or "ollama" (defaults to LLM_PROVIDER from env)
        model: Model name override
        api_key: API key override (Groq only)
        
    Returns:
        Initialized LLM client instance
        
    Example:
        # Use default config from .env
        client = get_llm_client()
        
        # Override for experiment
        client = get_llm_client(provider="groq", model="mixtral-8x7b-32768")
    """
    provider = (provider or LLM_PROVIDER).lower()
    
    if provider == "groq":
        return GroqClient(api_key=api_key, model=model)
    elif provider == "ollama":
        return OllamaClient(model=model)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Use 'groq' or 'ollama'."
        )


def generate_rag_response(
    question: str,
    context: str,
    client: BaseLLMClient = None,
    system_prompt: str = None,
    temperature: float = 0.1,
) -> Dict:
    """
    Generate a RAG response: answer a question using retrieved context.
    
    Args:
        question: The user's question
        context: Retrieved context from the vector store
        client: LLM client instance (creates default if None)
        system_prompt: Custom system prompt (defaults to RAG_SYSTEM_PROMPT)
        temperature: Sampling temperature
        
    Returns:
        Response dictionary from the LLM client
    """
    client = client or get_llm_client()
    system_prompt = system_prompt or RAG_SYSTEM_PROMPT
    
    user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )
    
    result = client.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    result["mode"] = "rag"
    result["question"] = question
    result["context_length"] = len(context)
    
    return result


def generate_baseline_response(
    question: str,
    client: BaseLLMClient = None,
    system_prompt: str = None,
    temperature: float = 0.1,
) -> Dict:
    """
    Generate a baseline response: answer a question without any context.
    
    This serves as the comparison baseline for the RAG approach.
    
    Args:
        question: The user's question
        client: LLM client instance (creates default if None)
        system_prompt: Custom system prompt (defaults to BASELINE_SYSTEM_PROMPT)
        temperature: Sampling temperature
        
    Returns:
        Response dictionary from the LLM client
    """
    client = client or get_llm_client()
    system_prompt = system_prompt or BASELINE_SYSTEM_PROMPT
    
    user_prompt = BASELINE_USER_PROMPT_TEMPLATE.format(question=question)
    
    result = client.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    result["mode"] = "baseline"
    result["question"] = question
    
    return result

"""LLM service for Gemini and OpenAI support."""
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import structlog

load_dotenv()
logger = structlog.get_logger()


class LLMService:
    """Unified LLM service supporting Gemini and OpenAI."""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """Initialize LLM service.
        
        Args:
            provider: LLM provider ("gemini" or "openai")
            model: Specific model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "gemini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            self.model_name = model or os.getenv("LLM_MODEL", "gemini-pro")
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("initialized_gemini", model=self.model_name)
            
        elif self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
            self.model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info("initialized_openai", model=self.model_name)
            
        elif self.provider == "groq":
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            self.client = Groq(api_key=api_key)
            self.model_name = model or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
            logger.info("initialized_groq", model=self.model_name)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using configured LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            if self.provider == "gemini":
                return await self._generate_gemini(prompt, system_prompt, **kwargs)
            elif self.provider == "openai":
                return await self._generate_openai(prompt, system_prompt, **kwargs)
            elif self.provider == "groq":
                return await self._generate_groq(prompt, system_prompt, **kwargs)
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e), provider=self.provider)
            raise
    
    async def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate using Gemini API."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate using OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
        
        return response.choices[0].message.content
    
    async def _generate_groq(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate using Groq API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
        
        return response.choices[0].message.content
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Synchronous generation for non-async contexts.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            if self.provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                
                generation_config = {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                return response.text
                
            elif self.provider == "openai":
                messages = []
                
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens)
                )
                
                return response.choices[0].message.content
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e), provider=self.provider)
            raise

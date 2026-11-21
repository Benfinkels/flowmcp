from abc import ABC, abstractmethod
import asyncio


class LLMBase(ABC):
    last_usage: dict = {}

    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass

    async def acomplete(self, prompt: str) -> str:
        return await asyncio.to_thread(self.complete, prompt)


class LLM:
    """
    Factory for creating LLM clients from a simple string.

    Supported formats:
        "openai/gpt-4"
        "anthropic/claude-3-5-sonnet"
        "ollama/llama3"
        "gemini/gemini-pro"
    """

    @staticmethod
    def from_string(model_string: str) -> LLMBase:
        if "/" not in model_string:
            raise ValueError(f"Invalid model format '{model_string}'. Use 'provider/model' e.g. 'openai/gpt-4'")

        provider, model = model_string.split("/", 1)

        if provider == "openai":
            from .openai import OpenAIClient
            return OpenAIClient(model)
        elif provider == "anthropic":
            from .anthropic import AnthropicClient
            return AnthropicClient(model)
        elif provider == "ollama":
            from .ollama import OllamaClient
            return OllamaClient(model)
        elif provider == "gemini":
            from .gemini import GeminiClient
            return GeminiClient(model)
        elif provider == "mistral":
            from .mistral import MistralClient
            return MistralClient(model)
        elif provider == "groq":
            from .groq import GroqClient
            return GroqClient(model)
        elif provider == "huggingface":
            from .huggingface import HuggingFaceClient
            return HuggingFaceClient(model)
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Supported: openai, anthropic, ollama, gemini, mistral, groq, huggingface")

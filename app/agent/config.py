"""Agent configuration — LLM factory and shared settings."""

from langchain_openai import AzureChatOpenAI
from app.core.config import settings


def get_llm(temperature: float = 0) -> AzureChatOpenAI:
    """Return a configured AzureChatOpenAI instance (gpt-5.4-nano)."""
    if not settings.AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY is not set")
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_API_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=temperature,
        timeout=settings.AGENT_TIMEOUT,
    )

"""Embedding and chat model factories for the configured provider."""

from src import config


def get_embeddings():
    config.require_keys(config.PROVIDER_KEY)
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL, output_dimensionality=config.EMBEDDING_DIM
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def get_chat_models() -> list:
    """One chat model per configured name, in fallback order."""
    config.require_keys(config.PROVIDER_KEY)
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Few retries so a model that is out of quota hands over to the next one quickly.
        return [ChatGoogleGenerativeAI(model=m, temperature=0, max_retries=1)
                for m in config.LLM_MODELS]
    from langchain_openai import ChatOpenAI

    return [ChatOpenAI(model=m, temperature=0) for m in config.LLM_MODELS]

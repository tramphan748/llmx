from ...utils import load_config
from .openai_textgen import OpenAITextGenerator
from .palm_textgen import PalmTextGenerator
from .cohere_textgen import CohereTextGenerator
from .anthropic_textgen import AnthropicTextGenerator
from .gemini_textgen import GeminiTextGenerator
import logging

logger = logging.getLogger("llmx")


def sanitize_provider(provider: str):
    if provider.lower() == "openai" or provider.lower() == "default" or provider.lower() == "azureopenai" or provider.lower() == "azureoai":
        return "openai"
    elif provider.lower() == "palm" or provider.lower() == "google":
        return "palm"
    elif provider.lower() == "gemini" or provider.lower() == "google":
        return "gemini"
    elif provider.lower() == "cohere":
        return "cohere"
    elif provider.lower() == "hf" or provider.lower() == "huggingface":
        return "hf"
    elif provider.lower() == "anthropic" or provider.lower() == "claude":
        return "anthropic"
    else:
        raise ValueError(
            f"Invalid provider '{provider}'. Supported providers are 'openai', 'hf', 'palm', 'gemini, 'cohere', and 'anthropic'."
        )


def llm(provider: str = None, **kwargs):

    # load config. This will load the default config from
    # configs/config.default.yml if no path to a config file is specified. in
    # the environment variable LLMX_CONFIG_PATH
    config = load_config()
    if provider is None:
        # provider is not explicitly specified, use the default provider from the config file
        provider = config["model"]["provider"] if "provider" in config["model"] else None
        kwargs = config["model"]["parameters"] if "parameters" in config["model"] else {}
    if provider is None:
        logger.info("No provider specified. Defaulting to 'openai'.")
        provider = "openai"

    # sanitize provider
    provider = sanitize_provider(provider)

    # set the list of available models based on the config file
    models = config["providers"][provider]["models"] if "providers" in config and provider in config["providers"] else {}

    kwargs["provider"] = kwargs["provider"] if "provider" in kwargs else provider
    kwargs["models"] = kwargs["models"] if "models" in kwargs else models

    # print(kwargs)

    if provider.lower() == "openai":
        return OpenAITextGenerator(**kwargs)
    elif provider.lower() == "palm":
        return PalmTextGenerator(**kwargs)
    elif provider.lower() == "gemini":
        return GeminiTextGenerator(**kwargs)
    elif provider.lower() == "cohere":
        return CohereTextGenerator(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicTextGenerator(**kwargs)
    elif provider.lower() == "hf":
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "Please install the `transformers` package to use the HFTextGenerator class. pip install llmx[transformers]"
            )

        # Check if torch package is installed
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Please install the `torch` package to use the HFTextGenerator class. pip install llmx[transformers]"
            )

        from .hf_textgen import HFTextGenerator

        return HFTextGenerator(**kwargs)

    else:
        raise ValueError(
            f"Invalid provider '{provider}'. Supported providers are 'openai', 'hf', 'palm', 'gemini', 'cohere', and 'anthropic'."
        )
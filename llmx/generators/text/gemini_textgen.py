from dataclasses import asdict
import os
import logging
from typing import Dict, Union
import google.generativeai as genai
from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import (
    cache_request,
    get_models_maxtoken_dict,
    num_tokens_from_messages,
)

logger = logging.getLogger("llmx")

class GeminiTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("GOOGLE_API_KEY", None),
        project_id: str = os.environ.get("GOOGLE_PROJECT_ID", None),
        provider: str = "gemini",
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)

        if api_key is None:
            raise ValueError("GOOGLE_API_KEY must be set.")
            
        self.api_key = api_key
        self.project_id = project_id
        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.model_name = model or "gemini-1.5-flash"
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def format_messages(self, messages):
        gemini_messages = []
        system_content = ""
        
        for message in messages:
            if message["role"] == "system":
                system_content += message["content"] + "\n"
            else:
                gemini_message = {
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": [{"text": message["content"]}]
                }
                gemini_messages.append(gemini_message)
                
        if system_content:
            # Prepend system message to first user message
            if gemini_messages and gemini_messages[0]["role"] == "user":
                gemini_messages[0]["parts"][0]["text"] = system_content + "\n" + gemini_messages[0]["parts"][0]["text"]
            else:
                gemini_messages.insert(0, {
                    "role": "user",
                    "parts": [{"text": system_content}]
                })

        return system_content, gemini_messages

    def generate(
        self,
        messages: Union[list[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name

        system_content, formatted_messages = self.format_messages(messages)
        self.model_name = model

        max_tokens = self.model_max_token_dict[model] if model in self.model_max_token_dict else 30720
        
        generation_config = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens or max_tokens,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "candidate_count": config.n
        }

        cache_key_params = {
            "messages": formatted_messages,
            "model": model,
            "generation_config": generation_config
        }

        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        try:
            chat = self.model.start_chat(history=formatted_messages)
            gemini_response = chat.send_message(
                formatted_messages[-1]["parts"][0]["text"],
                generation_config=generation_config
            )

            candidates = []
            for candidate in gemini_response.candidates:
                candidates.append({
                    "role": "assistant",
                    "content": candidate.content.parts[0].text
                })

            response_text = [Message(**x) for x in candidates]

            response = TextGenerationResponse(
                text=response_text,
                logprobs=[],
                config=generation_config,
                usage={
                    "total_tokens": num_tokens_from_messages(
                        response_text, model=self.model_name
                    )
                },
                response=gemini_response
            )

            cache_request(
                cache=self.cache, params=cache_key_params, values=asdict(response)
            )
            return response

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            raise

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
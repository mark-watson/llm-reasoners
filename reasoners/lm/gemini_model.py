import os
import time
from typing import Optional, Union

import numpy as np
from google import genai
from google.genai import types

from .. import LanguageModel, GenerateOutput

# Kept for backwards compatibility with code that reads this constant.
GEMINI_KEY = os.getenv("GEMINI_KEY", None)  # user needs to set the environment variable GEMINI_KEY

PROMPT_TEMPLATE_ANSWER = 'Your response need to be ended with "So the answer is"\n\n'
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"

# The key is resolved when the model is constructed, never at import time:
# `genai.Client()` raises when no key is available, so doing it at module level
# would make `import reasoners.lm` fail for every user who has no Gemini
# credentials.  This project runs against local oMLX models (see AGENTS.md).
_API_KEY_ENV_VARS = ("GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")

# Block nothing, matching the previous google-generativeai behaviour.
_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


def _resolve_api_key() -> Optional[str]:
    for name in _API_KEY_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    return None


class BardCompletionModel(LanguageModel):
    """Gemini backend built on the current ``google-genai`` SDK.

    The class name and public signature are unchanged; only the underlying SDK
    moved (``google-generativeai`` is deprecated and end-of-life).
    """

    def __init__(self, model: str, max_tokens: int = 2048, temperature=0.0, additional_prompt=None):
        api_key = _resolve_api_key()
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Set one of "
                f"{', '.join(_API_KEY_ENV_VARS)} before constructing BardCompletionModel."
            )
        self.model_name = model
        self.client = genai.Client(api_key=api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_prompt = additional_prompt

    def generate(self,
                 prompt: Optional[Union[str, list[str]]],
                 max_tokens: int = None,
                 rate_limit_per_min: Optional[int] = 60,
                 temperature: float = None,
                 additional_prompt=None,
                 retry=64,
                 **kwargs) -> GenerateOutput:

        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]

        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")

        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            safety_settings=_SAFETY_SETTINGS,
        )

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                return GenerateOutput(
                    text=[response.text],
                    log_prob=None
                )

            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        raise RuntimeError(f"BardCompletionModel failed to generate output, even after {retry} tries")

    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:

        raise NotImplementedError("BardCompletionModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                    prompt: Union[str, list[str]],
                    **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("BardCompletionModel does not support get_log_prob")

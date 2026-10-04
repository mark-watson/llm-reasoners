"""oMLX backend.

oMLX is a local, OpenAI-compatible multi-model inference server for Apple
silicon (MLX).  This project runs its experiments against oMLX *exclusively* --
see AGENTS.md.

Server settings live in ``~/.omlx/settings.json``.  The API key must be passed
via the ``OMLX_API_KEY`` environment variable; never commit it::

    export OMLX_API_KEY=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key'])")

Usage::

    from reasoners.lm import OMLXModel
    llm = OMLXModel()                       # project default (Laguna-XS-2.1-6bit)
    llm = OMLXModel("gemma")                # gemma-4 (manual chat template applied)
    llm = OMLXModel("mlx-community--Qwen3.8-27B-OptiQ-4bit")

Smoke-test the server and every registered model::

    python -m reasoners.lm.omlx_model
"""

import json
import os
import time
import urllib.request
from typing import Optional

from reasoners.base import GenerateOutput

from .openai_model import (
    PROMPT_TEMPLATE_ANSWER,
    PROMPT_TEMPLATE_CONTINUE,
    OpenAIModel,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# Project default model (per user preference, see AGENTS.md).
# Laguna is the fastest of the served models and the only one smoke-tested on
# more than one example (CoT + RAP).  gemma-4 is kept as an alias but needs the
# manual chat template below.
DEFAULT_MODEL = "Laguna-XS-2.1-6bit"

# Friendly aliases -> exact model ids advertised by `GET /v1/models`.
MODEL_ALIASES = {
    "default": DEFAULT_MODEL,
    "laguna": DEFAULT_MODEL,
    "laguna-xs": DEFAULT_MODEL,
    "gemma": "mlx-community--gemma-4-26b-a4b-6bit",
    "gemma-4": "mlx-community--gemma-4-26b-a4b-6bit",
    "qwen": "mlx-community--Qwen3.8-27B-OptiQ-4bit",
    "qwen3.8": "mlx-community--Qwen3.8-27B-OptiQ-4bit",
    "qwen-coder": "lmstudio-community--Qwen3-Coder-Next-MLX-4bit",
    "qwen3.6": "mlx-community--Qwen3.6-35B-A3B-4bit",
}

# Some MLX model repos ship no `chat_template` in tokenizer_config.json, so
# oMLX cannot format conversations for them: a Gemma-4 chat request comes back
# with empty content, and /v1/completions just continues the raw prompt.  For
# those models we apply the template ourselves and call the raw completions
# endpoint.  Verified: gemma-4 answers correctly this way.
CHAT_TEMPLATES = {
    "gemma": {
        "format": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "stop": ["<end_of_turn>"],
    },
}

# Substrings that select a manual template in chat_template="auto".
_AUTO_TEMPLATE_KEYS = (("gemma", "gemma"),)


def base_url() -> str:
    return os.getenv("OMLX_BASE_URL", DEFAULT_BASE_URL)


def api_key() -> str:
    """API key for the oMLX server.

    Falls back to reading it out of the oMLX settings file so that a local
    experiment runs without extra setup.
    """
    key = os.getenv("OMLX_API_KEY")
    if key:
        return key
    try:
        path = os.path.expanduser("~/.omlx/settings.json")
        with open(path) as f:
            return json.load(f)["auth"]["api_key"]
    except Exception:
        # oMLX accepts any key when skip_api_key_verification is on.
        return "omlx-local"


def resolve_model(model: str = None) -> str:
    """Map a friendly alias to a full oMLX model id."""
    if model is None:
        return DEFAULT_MODEL
    return MODEL_ALIASES.get(model.lower(), model)


def list_models() -> list:
    """Model ids currently served by oMLX."""
    req = urllib.request.Request(
        f"{base_url()}/models",
        headers={"Authorization": f"Bearer {api_key()}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return [m["id"] for m in json.load(resp)["data"]]


def unload(model_id: str) -> bool:
    """Evict one model from oMLX's in-memory pool.

    oMLX keeps models resident (LRU), and its memory guard will reject a 26-27B
    model while another large model is warm.  Evicting first avoids that.
    """
    req = urllib.request.Request(
        f"{base_url()}/models/{model_id}/unload",
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status == 200
    except Exception:
        return False


def unload_all() -> None:
    try:
        served = list_models()
    except Exception:
        return
    for model_id in served:
        unload(model_id)


class OMLXModel(OpenAIModel):
    """LanguageModel backed by a local oMLX server."""

    def __init__(
        self,
        model: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        additional_prompt=None,
        is_instruct_model: bool = True,
        chat_template: str = "auto",
    ):
        super().__init__(
            model=resolve_model(model),
            max_tokens=max_tokens,
            temperature=temperature,
            additional_prompt=additional_prompt,
            backend="omlx",
            is_instruct_model=is_instruct_model,
        )
        # "auto": apply a manual template only for models known to need one.
        # "none"/None: always use the server's chat endpoint.
        # a key of CHAT_TEMPLATES, or a literal "{prompt}" format string.
        self.chat_template = chat_template

    # ------------------------------------------------------------------ templates

    def _template(self) -> Optional[dict]:
        name = self.chat_template
        if name in (None, "", "none", "None"):
            return None
        if name == "auto":
            lowered = self.model.lower()
            for key, template_name in _AUTO_TEMPLATE_KEYS:
                if key in lowered:
                    return CHAT_TEMPLATES[template_name]
            return None
        if name in CHAT_TEMPLATES:
            return CHAT_TEMPLATES[name]
        return {"format": name, "stop": []}

    def _apply_additional_prompt(self, prompt: str, additional_prompt) -> str:
        if additional_prompt is None:
            additional_prompt = self.additional_prompt
        if additional_prompt == "ANSWER":
            return PROMPT_TEMPLATE_ANSWER + prompt
        if additional_prompt == "CONTINUE":
            return PROMPT_TEMPLATE_CONTINUE + prompt
        return prompt

    def _raw_completion(self, prompt, max_tokens, temperature, top_p, stop, retry=8):
        last_error = None
        for attempt in range(1, retry + 1):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                )
                return response.choices[0].text
            except Exception as e:  # noqa: BLE001 - retry the same way as the base class
                last_error = e
                print(f"[omlx] completion error: {e}; retry {attempt}/{retry}")
                time.sleep(min(attempt, 5))
        raise RuntimeError(
            f"oMLX completion failed after {retry} tries: {last_error}"
        )

    def _generate_with_template(
        self, prompts, template, num_return_sequences, stop,
        max_tokens, temperature, top_p, additional_prompt,
    ) -> GenerateOutput:
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        stops = list(template.get("stop") or [])
        if stop:
            stops.append(stop)
        texts = []
        for prompt in prompts:
            formatted = template["format"].format(
                prompt=self._apply_additional_prompt(prompt, additional_prompt)
            )
            for _ in range(max(num_return_sequences, 1)):
                texts.append(
                    self._raw_completion(
                        formatted, max_tokens, temperature, top_p, stops or None
                    )
                )
        return GenerateOutput(text=texts, log_prob=None)

    # ------------------------------------------------------------------ generate

    def generate(
        self,
        prompt=None,
        *args,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = None,
        stop: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        additional_prompt=None,
        **kwargs,
    ):
        """Generate, working around three oMLX limitations.

        1. oMLX accepts a single prompt per request, and
        2. it silently ignores the OpenAI ``n`` (num_return_sequences) parameter.

        Both are therefore emulated with sequential requests.  Independent
        sampling makes this distribution-equivalent to a batched call, just
        without server-side batching (slower for large ``n``).  The caller-visible
        count and ordering of returned texts is preserved.

        3. Models without a chat template (e.g. gemma-4) are driven through the
        raw completions endpoint with a manually applied template.

        ``rate_limit_per_min`` defaults to None: the inherited default of 20/min
        would sleep 3 s before every request, which is pointless on localhost.
        """
        prompts = prompt if isinstance(prompt, list) else [prompt]

        # HF-style callers express a stop token as `eos_token_id`; map it onto
        # the OpenAI `stop` parameter (the base class drops it).
        if stop is None:
            eos = kwargs.pop("eos_token_id", None)
            if isinstance(eos, str) and eos:
                stop = eos

        template = self._template()
        if template is not None:
            return self._generate_with_template(
                prompts, template, num_return_sequences, stop,
                max_tokens, temperature, top_p, additional_prompt,
            )

        if len(prompts) == 1 and num_return_sequences == 1:
            return super().generate(
                prompts, *args, rate_limit_per_min=rate_limit_per_min,
                num_return_sequences=1, stop=stop, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
                additional_prompt=additional_prompt, **kwargs,
            )

        texts = []
        for single in prompts:
            for _ in range(num_return_sequences):
                out = super().generate(
                    [single], *args, rate_limit_per_min=rate_limit_per_min,
                    num_return_sequences=1, stop=stop, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p,
                    additional_prompt=additional_prompt, **kwargs,
                )
                texts.extend(out.text)
        return GenerateOutput(text=texts, log_prob=None)


if __name__ == "__main__":
    served = list_models()
    print(f"oMLX at {base_url()}")
    print(f"served models ({len(served)}):")
    for m in served:
        print(f"  - {m}")

    question = ("Natalia sold clips to 48 friends in April, and then she sold half "
                "as many clips in May. How many clips did Natalia sell altogether "
                "in April and May? Give just the number.")
    for alias in ("default", "gemma", "qwen"):
        model_id = resolve_model(alias)
        missing = "" if model_id in served else "  [NOT SERVED]"
        print(f"\n--- {alias} -> {model_id}{missing}")
        # Free the pool first: oMLX's memory guard rejects a big model while
        # another large model is still resident.
        unload_all()
        try:
            llm = OMLXModel(model_id)
            out = llm.generate([question], max_tokens=64, temperature=0)
            print(f"  template: {llm._template() is not None}")
            print(f"  text: {out.text[0][:200]!r}")
        except Exception as e:  # noqa: BLE001 - report, keep going
            print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
    unload_all()

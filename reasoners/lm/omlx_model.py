"""oMLX backend.

oMLX is a local, OpenAI-compatible multi-model inference server for Apple
silicon (MLX).  This project runs its experiments against oMLX *exclusively* --
see AGENTS.md.

Server settings live in ``~/.omlx/settings.json``.  The API key must be passed
via the ``OMLX_API_KEY`` environment variable; never commit it::

    export OMLX_API_KEY=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key'])")

Usage::

    from reasoners.lm import OMLXModel
    llm = OMLXModel()                       # project default model
    llm = OMLXModel("laguna")               # alias for Laguna-XS-2.1-6bit
    llm = OMLXModel("mlx-community--Qwen3.8-27B-OptiQ-4bit")

Smoke-test the server and every registered model::

    python -m reasoners.lm.omlx_model
"""

import json
import os
import urllib.request

from .openai_model import OpenAIModel

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# Project default model (per user preference, see AGENTS.md).
DEFAULT_MODEL = "mlx-community--gemma-4-26b-a4b-6bit"

# Friendly aliases -> exact model ids advertised by `GET /v1/models`.
MODEL_ALIASES = {
    "default": DEFAULT_MODEL,
    "gemma": DEFAULT_MODEL,
    "gemma-4": DEFAULT_MODEL,
    "laguna": "Laguna-XS-2.1-6bit",
    "laguna-xs": "Laguna-XS-2.1-6bit",
    "qwen": "mlx-community--Qwen3.8-27B-OptiQ-4bit",
    "qwen3.8": "mlx-community--Qwen3.8-27B-OptiQ-4bit",
    "qwen-coder": "lmstudio-community--Qwen3-Coder-Next-MLX-4bit",
    "qwen3.6": "mlx-community--Qwen3.6-35B-A3B-4bit",
}


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


class OMLXModel(OpenAIModel):
    """LanguageModel backed by a local oMLX server."""

    def __init__(
        self,
        model: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        additional_prompt=None,
        is_instruct_model: bool = True,
    ):
        super().__init__(
            model=resolve_model(model),
            max_tokens=max_tokens,
            temperature=temperature,
            additional_prompt=additional_prompt,
            backend="omlx",
            is_instruct_model=is_instruct_model,
        )

    def generate(self, *args, rate_limit_per_min=None, **kwargs):
        # Local server: never throttle by default (base default is 20/min,
        # which would sleep 3s before every request).
        return super().generate(
            *args, rate_limit_per_min=rate_limit_per_min, **kwargs
        )


if __name__ == "__main__":
    served = list_models()
    print(f"oMLX at {base_url()}")
    print(f"served models ({len(served)}):")
    for m in served:
        print(f"  - {m}")

    for alias in ("default", "laguna", "qwen"):
        model_id = resolve_model(alias)
        missing = "" if model_id in served else "  [NOT SERVED]"
        print(f"\n--- {alias} -> {model_id}{missing}")
        try:
            llm = OMLXModel(model_id)
            out = llm.generate(["Reply with exactly: OK"], max_tokens=16, temperature=0)
            print(f"  text: {out.text[0][:200]!r}")
        except Exception as e:  # noqa: BLE001 - report, keep going
            print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")

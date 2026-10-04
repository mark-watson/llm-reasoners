# Project Instructions

## Git branch policy (IMPORTANT — permanent, do not override)

All work in this repository happens on the **`mw-experiments`** branch.

- **Always** work on `mw-experiments`.
- **Never** work on, commit to, or push to `main`. No exceptions.
- If the current branch is not `mw-experiments`, switch to it before making any
  changes:
  ```
  git switch mw-experiments
  ```
- Do not create new branches, and do not merge `mw-experiments` into `main`,
  unless the user explicitly asks for it in the current conversation.
- Never `git push` to `origin/main`.

If a task seems to require work on `main`, stop and ask the user first.

## Model backend policy (IMPORTANT — permanent, do not override)

This project is experimented on **exclusively with local models served by
oMLX** (MLX on Apple silicon). Do not route an experiment to a cloud API
(OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, …) and do not add a cloud
model as a fallback, unless the user explicitly asks for it in the current
conversation.

### Server

| | |
|---|---|
| Endpoint | `http://127.0.0.1:8000/v1` (OpenAI-compatible) |
| Settings | `~/.omlx/settings.json` |
| Lifecycle | `omlx start` / `omlx stop` / `omlx restart` |
| Health check | `curl -s http://127.0.0.1:8000/health` |
| List models | `curl -s http://127.0.0.1:8000/v1/models` |

Auth: oMLX expects a bearer token. The key lives in `~/.omlx/settings.json`
under `auth.api_key` — export it as `OMLX_API_KEY` and **never commit it**
(`.env` is already gitignored). `OMLX_BASE_URL` overrides the endpoint.

```bash
export OMLX_API_KEY=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key'])")
```

### Models

| role | model id |
|---|---|
| **default** | `mlx-community--gemma-4-26b-a4b-6bit` |
| alternate | `Laguna-XS-2.1-6bit` |
| alternate | `mlx-community--Qwen3.8-27B-OptiQ-4bit` |
| also served | `mlx-community--Qwen3.6-35B-A3B-4bit`, `lmstudio-community--Qwen3-Coder-Next-MLX-4bit`, `mlx-community--Laguna-XS-2.1-4bit` |

Use the project default unless the user names another model for the task at
hand. Always take ids from `GET /v1/models` rather than guessing, since oMLX
names local models with `--` in place of `/`.

### Using it from code

```python
from reasoners.lm import OMLXModel

llm = OMLXModel()                    # project default (gemma-4)
llm = OMLXModel("laguna")            # friendly alias
llm = OMLXModel("mlx-community--Qwen3.8-27B-OptiQ-4bit")
out = llm.generate(["..."])          # -> GenerateOutput(text=[...])
```

- Implementation: `reasoners/lm/omlx_model.py`; `backend="omlx"` is handled in
  `reasoners/lm/openai_model.py.__init_client__`.
- Friendly aliases live in `MODEL_ALIASES` (`reasoners/lm/omlx_model.py`).
  Add new models there instead of hardcoding ids in example scripts.
- Smoke-test the server and every registered model:
  `python -m reasoners.lm.omlx_model`
- When adding a new example/algorithm, expose `omlx` as a `base_lm` choice
  instead of `openai`/`anthropic`/`google`.

### Caveats

- **Memory guard.** oMLX keeps models resident (LRU pool). Loading a 26–27B
  model while another large model is warm can be rejected or aborted
  mid-prefill with `prefill_memory_exceeded` / `prefill_memory_aborted`.
  Check `curl -s http://127.0.0.1:8000/health` (`loaded_count`,
  `current_model_memory`) before blaming the code; `omlx restart` clears the
  pool. Do not silently switch models to dodge this — surface it to the user.
- **No logprobs.** `get_next_token_logits()` and `get_loglikelihood()` raise
  `NotImplementedError` on this backend, so algorithms that score candidates
  (self-eval / reward-model style) are not available through oMLX as-is.
- Requesting a model that is not in `GET /v1/models` fails at request time, not
  import time.

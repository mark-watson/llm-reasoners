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
| Unload a model | `POST /v1/models/{model_id}/unload` |
| Full API schema | `curl -s http://127.0.0.1:8000/openapi.json` |

Auth: oMLX expects a bearer token (any token, while
`auth.skip_api_key_verification` is true). The real key lives in
`~/.omlx/settings.json` under `auth.api_key` — **never commit it** (`.env` is
gitignored). `OMLX_API_KEY` and `OMLX_BASE_URL` override both.

```bash
export OMLX_API_KEY=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key'])")
```

### Models

| role | model id | usable? |
|---|---|---|
| **default** | `Laguna-XS-2.1-6bit` | yes, natively — fastest served model, smoke-tested on CoT + RAP |
| alias `gemma` | `mlx-community--gemma-4-26b-a4b-6bit` | yes, **only via manual chat template** |
| alias `qwen3.8` | `mlx-community--Qwen3.8-27B-OptiQ-4bit` | yes, natively, but ~18x slower end-to-end |
| also served | `mlx-community--Qwen3.6-35B-A3B-4bit`, `lmstudio-community--Qwen3-Coder-Next-MLX-4bit`, `mlx-community--Laguna-XS-2.1-4bit` | untested |

Measured on the same 8-example CoT/GSM8K run (2026-10-04): Laguna 78 s / 77.6
tok/s / 28.5 GB, gemma-4 56 s / 62.6 tok/s / 22.9 GB, Qwen3.8 1420 s / 13.7
tok/s / 20.4 GB. Laguna was chosen as the default on that basis; the accuracy
differences at n=8 are noise, so treat speed and robustness as the real signal.

Use the project default unless the user names another model for the task at
hand. Always take ids from `GET /v1/models` rather than guessing, since oMLX
names local models with `--` in place of `/`.

### Model gotchas (all verified against the live server)

- **gemma-4 ships no chat template.** Its `tokenizer_config.json` has no
  `chat_template`, so oMLX's `/v1/chat/completions` returns **empty content**
  (and raw completions just parrot the prompt). `OMLXModel` therefore applies a
  Gemma turn template itself and drives `/v1/completions`. If gemma-4 output
  ever looks empty or like prompt echo, check this first.
- **`n` (num_return_sequences) is ignored** — oMLX returns exactly one choice.
- **One prompt per request** is accepted; no batch inference endpoint.
- `OMLXModel` emulates both of the above with sequential requests so the
  caller-visible behaviour matches the other backends.
- **No log-probabilities.** `logprobs`/`top_logprobs` are accepted and silently
  dropped; there is no logits endpoint. So `get_next_token_logits()` and
  `get_loglikelihood()` still raise `NotImplementedError`. Any algorithm needing
  candidate scores must use a sampling-based reward instead.
- **Memory guard.** oMLX keeps models resident (LRU) and its guard rejects a
  26–27B model while another large model is warm (`prefill_memory_exceeded` /
  `prefill_memory_aborted`). Call `unload_all()` (or `POST
  /v1/models/{id}/unload`) before loading a second large model — restarting
  oMLX also clears the pool. Do not silently swap models to dodge this; surface
  it to the user.

### Using it from code

```python
from reasoners.lm import OMLXModel

llm = OMLXModel()                    # project default (Laguna-XS-2.1-6bit)
llm = OMLXModel("gemma")             # gemma-4: manual Gemma chat template applied
llm = OMLXModel("mlx-community--Qwen3.8-27B-OptiQ-4bit")
llm = OMLXModel(chat_template="none")  # force the server's chat endpoint

out = llm.generate(["..."])          # -> GenerateOutput(text=[...])
```

- Implementation: `reasoners/lm/omlx_model.py`; `backend="omlx"` is handled in
  `reasoners/lm/openai_model.py.__init_client__`.
- Friendly aliases live in `MODEL_ALIASES`; add new models there rather than
  hardcoding ids in example scripts. Manual chat templates live in
  `CHAT_TEMPLATES` (`chat_template="auto"` selects one by model id).
- Helpers: `list_models()`, `unload(id)`, `unload_all()`.
- Smoke-test the server and every registered model (unloads between models to
  avoid the memory guard): `python -m reasoners.lm.omlx_model`

### Running the examples

The environment is a `uv` venv in `.venv` (Python 3.11), with `torch`,
`transformers`, `datasets`, `peft`, `accelerate`, `fairscale`, `openai`,
`anthropic`, `google-genai`, `tarski` and `pddl==0.2.0` installed.
`bitsandbytes`, `optimum` and `ninja` are **not** installed (CUDA-oriented and
unused by these three). Recreate with::

    UV_CACHE_DIR=$PWD/.uv-cache uv venv --python 3.11 .venv
    UV_CACHE_DIR=$PWD/.uv-cache uv pip install --python .venv/bin/python \
      tqdm fire numpy scipy pandas sympy torch transformers datasets \
      huggingface_hub sentencepiece openai peft accelerate fairscale \
      anthropic google-genai pyyaml requests tarski pddl==0.2.0
    UV_CACHE_DIR=$PWD/.uv-cache uv pip install --python .venv/bin/python -e . --no-deps

Always run with `HF_HOME=$PWD/.hf-cache` (see below).

```bash
# chain-of-thought (no logprobs needed)
.venv/bin/python examples/CoT/gsm8k/inference.py --base_lm omlx --model_dir laguna --num_examples 2

# tree-of-thoughts / game24 (its default reward is sampling-based)
.venv/bin/python examples/ToT/game24/inference.py --base_lm omlx --omlx_model laguna --num_examples 2

# RAP (needs the sampling usefulness reward, selected automatically)
.venv/bin/python examples/RAP/gsm8k/inference.py --base_lm omlx --omlx_model laguna --num_examples 1
```

- `--num_examples N` (added to `Evaluator.evaluate`) limits a run; the full
  GSM8K test set is 1319 items and is impractical on a local model.
- **RAP + oMLX**: RAP's usefulness reward normally needs next-token log-odds.
  oMLX has none, so `calc_useful='sampling'` (n=`n_useful_samples`) is selected
  automatically with a printed warning. Those rewards are an **approximation**,
  not the original reward — say so when reporting results.
- **ToT + oMLX**: keep `calc_reward='sampling'`; `'logits'` raises a clear error.
- **ToT format caveat (measured 2026-10-04)**: none of the three models follow
  game24's strict `...(left: ...)` continuation format. `Laguna-XS-2.1-6bit`
  reasons with blank lines, so `get_actions` truncates at the first `\n\n` and
  returns **zero** actions (run scores 0/1, no tree edges grow).
  `Qwen3.8-27B-OptiQ-4bit` instead **parrots the in-prompt examples**, yielding
  syntactically valid but semantically wrong actions (`8 10 14` for input
  `4 5 6 10`) — verify actions against the actual input before trusting scores.
  `chat_template_kwargs={'enable_thinking': False}` did not help; the model then
  answers in prose. ToT needs a model that will continue a bare format.
- Batch size: `batch_size=1` is the safe choice for multi-prompt code paths.
- `reasoners/benchmark/gsm8k.py` uses the canonical `openai/gsm8k` dataset id;
  the bare `gsm8k` alias is rejected by `huggingface_hub >= 1.0`.
- Set `HF_HOME` to a workspace path (e.g. `$PWD/.hf-cache`), otherwise dataset
  downloads fail: the default `~/.cache/huggingface` is outside the writable
  workspace.
- Set `XDG_CACHE_HOME` to a workspace path (e.g. `$PWD/.cl-cache`) when running
  SBCL here, or ASDF cannot write its fasl cache under `~/.cache/common-lisp`.

## Common Lisp port (`common-lisp/`)

`common-lisp/llm-reasoning-lib.lisp` is a small port of the core llm-reasoners
abstractions (language model / world model / search config / beam search /
reasoner, plus chain-of-thought with self-consistency) on top of the locally
installed **litelm** Quicklisp library. `common-lisp/CoT-gsm8k-example.lisp`
answers GSM8K with it through oMLX.

```bash
sbcl --script common-lisp/CoT-gsm8k-example.lisp        # 6/6 with Laguna
sbcl --script common-lisp/CoT-gsm8k-example.lisp 6 5    # 5-sample self-consistency
sbcl --non-interactive --load common-lisp/llm-reasoning-lib.lisp \
     --eval '(llm-reasoning:run-self-tests)'            # 17 offline checks
```

- litelm routes on a `"provider/model-name"` string; `omlx/Laguna-XS-2.1-6bit`
  is its default and needs no API key. Do **not** use gemma-4 through litelm —
  that MLX repo has no chat template, so oMLX returns empty content (the Python
  `OMLXModel` works around it; litelm does not).
- **Known environment bug — dexador + `*PRINT-CASE*`.** litelm POSTs through
  dexador. dexador's `DEFINE-ALIST-CACHE` builds function names with
  `(FORMAT NIL "LOOKUP-IN-~A" ...)`, which follows `*PRINT-CASE*`. With the
  `(setf *print-case* :downcase)` in `~/.sbclrc`, dexador compiles definitions
  named `|LOOKUP-IN-content-encoding-cache|` while the source references read as
  `LOOKUP-IN-CONTENT-ENCODING-CACHE`, so **every POST that carries a body fails**
  with "The function ... is undefined" — while GET keeps working, which makes it
  look like anything but a name-case bug. One-time fix:

  ```lisp
  (let ((*print-case* :upcase)) (asdf:load-system :dexador :force t))
  ```

  The library binds `*PRINT-CASE*` to `:UPCASE` around its own litelm load so a
  fresh compile is correct; an already-cached bad fasl still needs the force
  above.
- `llm-reasoning` shadows the `COMMON-LISP` symbols `STEP` and `SEARCH`, so a
  package that `:USE`s both CL and `llm-reasoning` must shadowing-import them.
  `CoT-gsm8k-example.lisp` imports only the symbols it needs instead.
- `sbcl --script` skips `~/.sbclrc`, so Quicklisp and the `litelm.asd` project
  directory are both missing; the library's bootstrap handles that itself
  (honouring `LITELM_ASD` if litelm lives somewhere unusual).
- Measured: CoT on the first 6 GSM8K test questions scores 6/6 with
  `Laguna-XS-2.1-6bit`, matching the Python example. Two settings matter — the
  `additional_prompt="ANSWER"` instruction, and a 2048-token budget (at 1024 the
  150%-profit question is truncated mid-deliberation and scores as wrong).


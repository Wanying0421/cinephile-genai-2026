# Required model setup

The instructor requires `gpt-6-luna` for all language-model calls, including the
evaluation judge. The local multilingual-e5 embedding model is used for retrieval
and is not the answer-generation model.

## Local configuration (every team member)

If `.env` does not exist, copy `.env.example` to `.env`. If it already exists,
edit the existing file so other settings are preserved. Configure:

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=PASTE_THE_AUTHORISED_KEY_LOCALLY
LLM_MODEL=gpt-6-luna
RAG_MIN_SCORE=
```

Use the instructor's endpoint if one is supplied instead of the direct OpenAI
endpoint. `.env` is ignored by Git. Never put a real key in `.env.example`, a
commit, a screenshot or a shared trace. Each member updates their local file.
Existing shell environment variables take precedence over `.env`; update them
if they still select the old provider or model. Restart Python after editing.

## API compatibility

`level0.complete()` centralises Chat Completions requests for Level 0, the RAG
generator and the evaluation judge. It requires `gpt-6-luna`, defaults to the
OpenAI endpoint and explicitly sets `reasoning_effort="none"` and `temperature=0`.
The reasoning setting keeps temperature compatible with GPT-6 Luna and suits the
current short-answer workload. The future agent should review tool-call settings
when it is implemented.

Official references:
- https://developers.openai.com/api/docs/models/gpt-6-luna
- https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra#migration-quickstart

## Windows environment used for validation

The global Windows installation blocked a PyTorch DLL with App Control event
3077. A fresh project-local environment with the official CPU package loaded
successfully (`torch 2.15.0+cpu`). These commands install into `.venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install --upgrade typing_extensions --only-binary=:all: --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -c "import torch, numpy, faiss; from sentence_transformers import SentenceTransformer; print('Dependencies OK')"
.\.venv\Scripts\python.exe check_llm.py
```

Use `.\.venv\Scripts\python.exe` for subsequent commands, including integration
tests. Existing indexes and the Hugging Face model cache can be reused. If an
application-control policy also blocks the new installation, consult the device
administrator rather than assuming a package reinstall will resolve the policy.

## Checks before publishing new results

From the repository root, run one short connection check:

```powershell
python check_llm.py
```

This makes one API request. It prints the configured model, returned model and
an OK response, without printing the key. A successful check confirms API access;
it does not evaluate answer quality.

The retrieval index can be reused; changing the generation model does not change
the embedding model, chunking or corpus. If no local index exists, build it with
`python -m retrieval.build_index`.

Then rerun the generation integration cases with the real retriever: a cited
director answer, an information-absence question, and the difficult wristwatch
question. Inspect the actual outcomes, including failures; do not assume they
will match the earlier model. Save new traces with the configured model,
question, answer, citations, abstention reason, usage and latency.

Run the current Level 0 evaluator with:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
python eval.py tests/tests.jsonl
```

The current evaluator runs Level 0 only. The full Level 0/1 comparison remains
part of the evaluation work; this migration does not add that feature. Preserve
old traces before rerunning because the evaluator replaces
`results/level0_results.jsonl`. The judge uses the same required model.

## Reporting

On November 15, 2026, the Windows environment above passed dependency imports
and the connection check. Three real-retriever RAG requests completed with the
required model: the director answer was accepted, the music question correctly
abstained, and the difficult wristwatch question still failed by abstaining.
See [the generation handoff](level1-generation.md) for the measurements. This
validates the migration smoke checks; it is not the full Level 0/1 evaluation.

The Groq results in the README and generation handoff are historical development
runs. Retain their provider/model labels. Only publish new GPT-6 Luna quality,
token and latency measurements after real calls have completed. Review prompts
and abstention behaviour using the new results before comparing levels.

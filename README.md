# Cinéphile GenAI 2026

A movie and TV assistant built in levels, each one measured on the same test set:
**Level 0** LLM alone → **Level 1** RAG → **Level 2** tool-using agent → **Level 3** extension.

Project: Mineur IA Générative 2026 (Text track). Theme: Cinéphile.

## Team

| Member | Role |
|---|---|
| Member 1 | Corpus and data (movie/TV summaries, TMDB API) |
| Member 2 | Test protocol (`tests/tests.jsonl`, metrics in `METRICS.md`) |
| Member 3 | Level 0 baseline, repository, evaluation script |

## Repository structure

```
data/            movie and TV corpus (English)
tests/           tests.jsonl (test cases)
level0.py        Level 0: LLM alone, no documents, no tools
eval.py          runs a level on the test set and prints the metrics
METRICS.md       definition of the metrics
results/         raw results (not committed)
.env.example     environment variables template
```

## Installation

```bash
git clone https://github.com/Wanying0421/cinephile-genai-2026.git
cd cinephile-genai-2026
pip install -r requirements.txt
```

Create your own `.env` file from the template and fill in your values:

```bash
cp .env.example .env
```

```
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_own_key_here
LLM_MODEL=openai/gpt-oss-120b
LLM_TEMPERATURE=0
TMDB_API_KEY=
```

**Never commit `.env`.** Each member uses their own API key.

## Usage

```bash
mkdir results
python eval.py tests/tests.jsonl
```

This runs Level 0 on every test case, writes raw results to
`results/level0_results.jsonl`, and prints the summary metrics.

## Level 0 design

- No documents, no tools: the model answers from memory only.
- The system prompt asks the model to start with `[UNCERTAIN]` when it is not sure,
  so abstention can be detected automatically.
- Scoring per category:
  - `rag_document`: LLM judge compares the answer to the expected output
  - `rag_absence`: correct if the model abstains
  - `ambiguous`: correct if the model abstains or asks a clarifying question
  - `multi_step`: not scored at Level 0 (no tools available)

## Results

| Level | Accuracy | Abstention | Tokens / request | Latency |
|---|---|---|---|---|
| 0 · LLM alone | 86% (6/7) | 50% (1/2) | 409 | 0.63 s |

Setup: 9 test cases (7 scored), model `openai/gpt-oss-120b` via Groq, temperature 0.

## Known limitations

- Only 9 test cases so far; the final set will have at least 40, so these numbers are provisional.
- The judge model is the same as the answering model, so accuracy may be optimistic.
  A 20-case human-annotated calibration is planned.
- Abstention detection only recognises the `[UNCERTAIN]` marker. In test 2, the model
  refused politely ("I can't help with that") without the marker, so it was scored as
  a failure. The true abstention rate is likely higher than reported.

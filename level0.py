import os
import time
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL = os.getenv("LLM_MODEL", "gpt-6-luna").strip() or "gpt-6-luna"
if MODEL != "gpt-6-luna":
    raise ValueError("This project requires LLM_MODEL=gpt-6-luna. Update your local .env and shell environment.")

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "").strip() or "https://api.openai.com/v1",
    api_key=os.getenv("LLM_API_KEY"),
)


def complete(messages):
    """Use the required model and compatible parameters for every LLM call."""
    return client.chat.completions.create(
        model=MODEL,
        reasoning_effort="none",
        temperature=0,
        messages=messages,
    )

SYSTEM_PROMPT = """You are an assistant specialized in movies and TV series.
Answer in English, concisely and factually.

Rules:
- Only answer if you are confident in the information.
- If you are not sure, or if you do not know the movie or series asked about,
  start your answer with [UNCERTAIN] and briefly explain what you don't know.
  Never invent a title, name, date, or plot.
- If the question is ambiguous, ask a clarifying question.
- If the question is unrelated to movies or TV series, politely refuse.
"""

def run(question):
    t0 = time.time()
    r = complete([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ])
    answer = r.choices[0].message.content or ""
    return {
        "level": 0,
        "question": question,
        "answer": answer,
        "abstained": answer.strip().startswith("[UNCERTAIN]"),
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
        "latency_s": round(time.time() - t0, 2),
        "model": MODEL,
    }

if __name__ == "__main__":
    print(json.dumps(run("Who directed Inception?"), indent=2, ensure_ascii=False))
    print(json.dumps(run("Who plays Mira Valdez in the series 'The Shadows of Lyon'?"), indent=2, ensure_ascii=False))

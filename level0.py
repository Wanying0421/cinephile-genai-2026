import os
import time
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
MODEL = os.getenv("LLM_MODEL")

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
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
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
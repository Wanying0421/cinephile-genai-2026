"""Download movie/TV texts from Wikipedia -> data/corpus_wiki.jsonl

Usage: python build_corpus.py
Input : data/titles.txt, one line per work: id|Wikipedia title|type|year
Output: same fields as data/corpus.jsonl (id, title, type, year, text, source_url)
Wikipedia text is licensed CC BY-SA 4.0.
"""
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
TITLES = ROOT / "data" / "titles.txt"
OUT = ROOT / "data" / "corpus_wiki.jsonl"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "CinephileRAG-student-project/0.1 (https://github.com/Wanying0421/cinephile-genai-2026)"}

# Keep the lead section + these sections; drop Reception, Box office, etc. (noise)
KEEP = ("plot", "synopsis", "premise", "cast", "summary")


def fetch(title, max_retries=5):
    """Return (real_title, plain_text) or (None, None) if the page does not exist."""
    params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
              "titles": title, "format": "json", "formatversion": 2}
    for attempt in range(max_retries):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            # Rate limited: wait as long as the server asks, else back off 5s, 10s, 20s...
            wait = int(r.headers.get("Retry-After", 5 * 2 ** attempt))
            print(f"  [429] rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        page = r.json()["query"]["pages"][0]
        if page.get("missing"):
            return None, None
        return page["title"], page.get("extract", "")
    raise RuntimeError(f"still rate limited after {max_retries} retries: {title}")


def keep_sections(text):
    # In plain-text extracts, headings look like "== Plot ==" or "=== Subsection ==="
    parts = re.split(r"\n(={2,})\s*(.+?)\s*\1\n", text)
    out = [parts[0].strip()]  # lead: director, year, main cast are usually here
    top = None
    for i in range(1, len(parts) - 2, 3):
        level, heading, body = parts[i], parts[i + 1], parts[i + 2]
        if len(level) == 2:          # a new top-level section starts
            top = heading.lower()
        if top and top.startswith(KEEP) and body.strip():
            out.append(f"{heading}\n{body.strip()}")
    return "\n\n".join(p for p in out if p)


def main():
    # utf-8-sig: ignores the invisible BOM some Windows editors add
    lines = TITLES.read_text(encoding="utf-8-sig").splitlines()
    entries = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split("|")]
        if len(fields) != 4:
            print(f"[BAD LINE] {line}")
            continue
        entries.append(fields)

    n_ok = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for doc_id, wiki_title, kind, year in entries:
            real_title, extract = fetch(wiki_title)
            if real_title is None:
                print(f"[MISSING]   {doc_id:<35} '{wiki_title}' not found")
                continue
            text = keep_sections(extract)
            n_words = len(text.split())
            if n_words < 100:
                print(f"[TOO SHORT] {doc_id:<35} {n_words} words")
                continue
            doc = {"id": doc_id, "title": real_title, "type": kind, "year": int(year),
                   "text": text,
                   "source_url": "https://en.wikipedia.org/wiki/" + real_title.replace(" ", "_")}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n_ok += 1
            print(f"[OK]        {doc_id:<35} {n_words} words")
            time.sleep(1.5)  # be polite to the Wikipedia API
    print(f"\nDone: {n_ok}/{len(entries)} documents -> {OUT}")


if __name__ == "__main__":
    main()

"""Load the corpus and split each document into overlapping word-window chunks."""
import json

from .config import CORPUS_PATH, CHUNK_SIZES, OVERLAP_RATIO


def load_documents(path=CORPUS_PATH):
    """Read corpus.jsonl and return a list of document dicts."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def chunk_document(doc, size, overlap):
    """Split one document into chunks of `size` words, each sharing `overlap`
    words with the previous chunk. Metadata is copied onto every chunk."""
    words = doc["text"].split()
    step = size - overlap

    # Slide a window of `size` words, moving forward by `step` words each time
    pieces = []
    start = 0
    while True:
        end = min(start + size, len(words))
        pieces.append(words[start:end])
        if end >= len(words):
            break
        start += step

    # A very short last piece carries little meaning on its own:
    # merge its new words (the part after the overlap) into the previous piece
    if len(pieces) > 1 and len(pieces[-1]) < size // 2:
        tail = pieces.pop()
        pieces[-1] = pieces[-1] + tail[overlap:]

    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append({
            "chunk_id": f"{doc['id']}#{i}",   # e.g. "inception-2010#3"
            "doc_id": doc["id"],
            "title": doc["title"],
            "type": doc["type"],
            "year": doc["year"],
            "source_url": doc["source_url"],
            "text": " ".join(piece),
        })
    return chunks


def chunk_corpus(docs, size):
    """Chunk every document with the given size; overlap comes from config."""
    overlap = int(size * OVERLAP_RATIO)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, size, overlap))
    return all_chunks


if __name__ == "__main__":
    # Quick sanity check: chunk counts per size and one example chunk
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    for size in CHUNK_SIZES:
        chunks = chunk_corpus(docs, size)
        avg = sum(len(c["text"].split()) for c in chunks) / len(chunks)
        print(f"size={size:<4} -> {len(chunks):>4} chunks, avg {avg:.0f} words")

    example = chunk_corpus(docs[:1], 200)
    print("\nFirst two chunks of", docs[0]["title"])
    for c in example[:2]:
        print(f"\n[{c['chunk_id']}] {c['text'][:300]}...")
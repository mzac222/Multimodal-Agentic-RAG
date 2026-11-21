import json
from pathlib import Path

import numpy as np
import ollama

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "docs"
INDEX_FILE = PROJECT_DIR / "index.json"

EMBED_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3-8k"

MAX_CHUNK_WORDS = 300
TOP_K = 4

QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages from personal notes that answer it properly in simple words\n"
    "Query: "
)


def load_documents():
    """Read every .md and .txt file in the docs folder as (filename, text) pairs."""
    if not DOCS_DIR.exists():
        print("Docs not found")
        return []

    docs = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
            docs.append((path.name, text))
    return docs


def chunk_document(filename, text):
    """Split a document into chunks by '## ' sections, capped at MAX_CHUNK_WORDS each."""
    chunks = []
    section = "Introduction"
    lines = []

    def save_section():
        """Turn the lines collected for the current section into one or more chunks."""
        body = "\n".join(lines).strip()
        if not body:
            return

        pieces = []
        current = ""
        for para in body.split("\n\n"):
            candidate = (current + "\n\n" + para).strip()
            if current and len(candidate.split()) > MAX_CHUNK_WORDS:
                pieces.append(current)
                current = para
            else:
                current = candidate
        if current:
            pieces.append(current)

        for piece in pieces:
            chunks.append({
                "source": filename,
                "section": section,
                "chunk_id": len(chunks),
                "text": f"{section}\n{piece}",
            })

    for line in text.splitlines():
        if line.startswith("## "):
            save_section()
            section = line[3:].strip()
            lines = []
        elif line.startswith("# "):
            continue
        else:
            lines.append(line)
    save_section()

    return chunks


def embed_chunks(chunks):
    """Add an embedding vector to every chunk."""
    texts = [c["text"] for c in chunks]
    resp = ollama.embed(model=EMBED_MODEL, input=texts)

    for chunk, vector in zip(chunks, resp["embeddings"]):
        chunk["vector"] = vector
    return chunks


def docs_snapshot():
    """Record each doc's size and last-modified time to detect changes."""
    snapshot = {}
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() in (".md", ".txt"):
            info = path.stat()
            snapshot[path.name] = [info.st_size, info.st_mtime]
    return snapshot


def save_index(chunks, snapshot):
    """Save the embedded chunks and docs snapshot to index.json."""
    records = []
    for c in chunks:
        records.append({
            "id": f"{c['source']}#{c['chunk_id']}",
            "vector": c["vector"],
            "text": c["text"],
            "metadata": {
                "source": c["source"],
                "section": c["section"],
                "chunk_id": c["chunk_id"],
            },
        })

    data = {"model": EMBED_MODEL, "snapshot": snapshot, "records": records}
    INDEX_FILE.write_text(json.dumps(data), encoding="utf-8")


def load_index(snapshot):
    """Return saved records from index.json, or None if missing or out of date."""
    if not INDEX_FILE.exists():
        return None

    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if data["snapshot"] != snapshot:
        return None
    if data["model"] != EMBED_MODEL:
        return None
    return data["records"]


def get_index():
    """Load the saved index if it's fresh, otherwise rebuild and save it."""
    snapshot = docs_snapshot()

    records = load_index(snapshot)
    if records is not None:
        print(f"Loaded {len(records)} chunks from {INDEX_FILE.name} (no changes found)")
        return records

    print("Building index...")
    chunks = []
    for name, text in load_documents():
        chunks.extend(chunk_document(name, text))
    chunks = embed_chunks(chunks)
    save_index(chunks, snapshot)
    print(f"Saved {len(chunks)} chunks to {INDEX_FILE.name}")

    return load_index(snapshot)


def search(question, records, k=TOP_K):
    """Return the k chunks most similar to the question, with their scores."""
    resp = ollama.embed(model=EMBED_MODEL, input=[QUERY_INSTRUCTION + question])
    q_vec = np.array(resp["embeddings"][0])
    matrix = np.array([r["vector"] for r in records])

    dot = matrix @ q_vec
    lengths = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec)
    scores = dot / lengths

    best = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), records[i]) for i in best]


def answer(question, results):
    """Stream an answer from the chat model using the retrieved chunks, then print citations."""
    context_parts = []
    for n, (score, r) in enumerate(results, start=1):
        label = f"[{n}] ({r['metadata']['source']} › {r['metadata']['section']})"
        context_parts.append(f"{label}\n{r['text']}")
    context = "\n\n".join(context_parts)

    prompt = (
        "Introduce yourself as fitbot and be kind and polite and explain in a joyful way.\n"
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say \"I don't know based on your notes.\"\n"
        "Cite the sources you used like [1] or [2].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    stream = ollama.chat(
        model=CHAT_MODEL,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
        think=False,
    )

    print("\nAnswer: ", end="")
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print("")

    print("Citations:")
    for n, (score, r) in enumerate(results, start=1):
        print(f"  [{n}] {r['metadata']['source']} › {r['metadata']['section']}  (score {score:.3f})")


if __name__ == "__main__":
    records = get_index()

    print("\nAsk a question (press Enter on an empty line to quit).")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            break

        results = search(question, records)
        answer(question, results)

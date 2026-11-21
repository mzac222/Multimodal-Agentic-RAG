from pathlib import Path
import ollama
import json 
import numpy as np
PROJECT_DIR = Path(__file__).parent      # the folder this script lives in
DOCS_DIR = PROJECT_DIR / "docs"          # where your notes live
INDEX_FILE = PROJECT_DIR / "index.json"  # our mini vector DB (used in step 4)

EMBED_MODEL = "qwen3-embedding:0.6b"     # turns text into 1024 numbers
CHAT_MODEL = "qwen3-8k"                  # writes the final answer

MAX_CHUNK_WORDS = 300
TOP_K = 4

QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages from personal notes that answer it properly in simple words introduce yourself as zekefit\n"
    "Query: "
)
def search(question, records, k=TOP_K):
    resp = ollama.embed(model=EMBED_MODEL,input=[QUERY_INSTRUCTION + question])





# ---------------------------------------------------------------------------
# Step 4: save / load the index (our mini vector DB)
# ---------------------------------------------------------------------------
def docs_snapshot():
    """
    A quick 'has anything changed?' fingerprint of the docs folder.
    Example: {"gym_notes.md": [1203, 1732119000.5], "cooking_notes.md": [1450, 1732119100.2]}
             (file size in bytes, last-modified time)
    """
    snapshot = {}
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() in (".md", ".txt"):
            info = path.stat()                          # file details from Windows
            snapshot[path.name] = [info.st_size, info.st_mtime]
    return snapshot


def save_index(chunks, snapshot):
    """Write every chunk to index.json in vector DB format: id, vector, text, metadata."""
    records = []
    for c in chunks:
        records.append({
            "id": f"{c['source']}#{c['chunk_id']}",     # e.g. "gym_notes.md#0"
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
    """
    Return the saved records if index.json exists AND nothing has changed.
    Otherwise return None, meaning 'please rebuild'.
    """
    if not INDEX_FILE.exists():
        return None                                     # first run: nothing saved yet

    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if data["snapshot"] != snapshot:
        return None                                     # a note was added, edited or deleted
    if data["model"] != EMBED_MODEL:
        return None                                     # different embedding model = different fingerprints
    return data["records"]


def get_index():
    """Load the index if it's still fresh, otherwise build it from scratch and save it."""
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

    return load_index(snapshot)                          # reload so we always return the same format

def load_documents():
    if not DOCS_DIR.exists():
        print("Docs not found")
        return []
    docs = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() in (".txt",".md"):
           text = path.read_text(encoding="utf-8")
           docs.append((path.name,text))
    return docs


def chunk_document(filename, text):
    chunks = []
    section = "Introduction"
    lines = []
    def save_section():
        body = "\n".join(lines).strip()
        if not body : 
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
                "source" : filename,
                "section" : section,
                "chunk_id" : len(chunks),
                "text" : f"{section}\n{piece}"
                        }
            )

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
    texts = [c["text"] for c in chunks]
    resp = ollama.embed(model= EMBED_MODEL,input=texts)
    vectors = resp["embeddings"]

    for chunk,vector in zip(chunks,vectors):
        chunk["vector"] = vector
    return chunks 





if __name__ == "__main__":
    records = get_index()

    print("\nAsk a question (press Enter on an empty line to quit).")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            break

        results = search(question, records)
        print("Top matches:")
        for score, r in results:
            print(f"  {score:.3f}  {r['id']}  [{r['metadata']['section']}]")
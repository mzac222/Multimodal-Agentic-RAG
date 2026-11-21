from pathlib import Path
from pypdf import PdfReader
from data import BOOKS
from qdrant_client import QdrantClient, models
import time 
import uuid 
import ollama


PROJECT_DIR = Path(__file__).parent
PDF_DIR = PROJECT_DIR / "docs" / "pdfs"     
CHUNK_WORDS = 300    
OVERLAP_WORDS = 50     # words shared with the previous chunk, so ideas aren't cut in half
QDRANT_DIR = PROJECT_DIR / "qdrant_data"
COLLECTION = "health_books"
EMBED_DIM = 1024 #qwen3-embedding:0.6b gives 102
EMBED_MODEL = "qwen3-embedding:0.6b"
BATCH_SIZE = 32
CHAT_MODEL = "qwen3-8k"
TOP_K = 4
MIN_SCORE = 0.32
TOPICS = ("workout", "nutrition")


# Router: decides what kind of message this is before we search
ROUTER_PROMPT = (
    "Classify the user's message into exactly ONE category:\n"
    "- smalltalk: greetings, thanks, chit-chat, questions about you\n"
    "- workout: exercise, training, warm-ups, fitness, recovery\n"
    "- nutrition: food, diet, protein, vitamins, calories, digestion\n"
    "- general: health questions that involve both exercise and nutrition\n"
    "Reply with only the category word."
)
ROUTES = ("smalltalk", "workout", "nutrition", "general")

QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages from health and fitness books that answer it\n"
    "Query: "
)

def load_pdfs():
    pages = []
    for path in sorted(PDF_DIR.glob("*.pdf")):
        reader = PdfReader(path)
        info = get_book_info(path.name)
        if info is None:
                print(f"WARNING: no catalogue entry for {path.name} - add it to BOOKS")
                info = {"title": path.stem, "topic": "unknown", "license": "unknown", "url": ""}
        chapters = get_chapters(reader)
        for page_num,page in enumerate(reader.pages,start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            if text : 
                pages.append({
                    "source": path.name,
                    "title": info["title"],
                    "topic": info["topic"],
                    "license": info["license"],
                    "url": info["url"],
                    "chapter": chapter_for_page(chapters, page_num),
                    "page": page_num,
                    "text": text,
                })
    return pages 


def get_book_info(filename):
    """Return the catalogue entry whose match-words appear in the file name, or None."""
    name = filename.lower()
    for book in BOOKS:
        if any(word in name for word in book["match"]):
            return book
    return None

def get_chapters(reader):
    chapters = []
    for item in reader.outline:
        if  isinstance(item,list) : 
            continue
        try:
            start = reader.get_destination_page_number(item) + 1 
        except Exception:
            continue
        chapters.append([start,item.title])
    chapters.sort()
    return chapters

def chapter_for_page(chapters, page_num):
    """Return the title of the chapter that contains this page."""
    current = "Unknown"
    for start, title in chapters:
        if start <= page_num:
            current = title               # this chapter started on or before our page
        else:
            break                         # chapters after this start later - stop
    return current

def split_words(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    pieces = []
    for start in range(0, len(words),step):
        pieces.append(" ".join(words[start:start+CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
    return pieces

def chunk_pages(pages):
    chunks = []
    for page in pages: 
        for n,piece in enumerate(split_words(page["text"]),start=1):
            chunks.append({
                "id": f"{page['source']}#p{page['page']}-c{n}",       # e.g. "H2F-FM-7-22-2020.pdf#p45-c2"
                "text": f"{page['title']} > {page['chapter']}\n{piece}",   # label line gives search extra context
                "metadata": {
                    "source": page["source"],
                    "title": page["title"],
                    "topic": page["topic"],
                    "license": page["license"],
                    "url": page["url"],
                    "chapter": page["chapter"],
                    "page": page["page"],
                    "chunk": n
                },
            })
    return chunks

def get_qdrant():
    client = QdrantClient(path = str(QDRANT_DIR))
    if not client.collection_exists(COLLECTION) : 
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )

        print(f"collection has been created {COLLECTION}")
    else:
        print("collection already exists")
    return client

def make_point_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL,chunk_id))

def ingest(client,chunks):
    total = len(chunks)
    start_time = time.time()

    for start in range(0,total,BATCH_SIZE):
        batch = chunks[start:start+BATCH_SIZE]
        text = [c["text"] for c in batch]
        vectors = ollama.embed(model=EMBED_MODEL, input= text)["embeddings"]

        points = []
        for chunk, vector in zip(batch,vectors):
            points.append(models.PointStruct(
                    id = make_point_id(chunk["id"]),
                    vector= vector,
                    payload={"chunk_id": chunk["id"], "text": chunk["text"], **chunk["metadata"]},  
            )
            
            )
        client.upsert(collection_name=COLLECTION, points=points)
        done = min(start + BATCH_SIZE, total)
        print(f"  stored {done:,} / {total:,} chunks", end="\r")   # \r = overwrite the same line

    print(f"\nDone in {time.time() - start_time:.0f} seconds")

def search(client, question , top_k,topic): 
    q_vec = ollama.embed(model=EMBED_MODEL, input=[QUERY_INSTRUCTION + question])["embeddings"][0]
    query_filter =None 
    if topic: 
        query_filter = models.Filter(
            must=[models.FieldCondition(key="topic", match=models.MatchValue(value=topic))]
        )

    response = client.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    )
    # turn Qdrant's results into (score, payload) pairs for answer()
    return [(point.score, point.payload) for point in response.points]

def cite(p):
    """Build a readable citation from a payload, e.g. 'Human Nutrition (2020 Edition) > Chapter 5. Lipids > p.364'"""
    return f"{p['title']} > {p['chapter']} > p.{p['page']}"

def answer(question, results):
    system_prompt = (
        "You are a friendly fitness and nutrition assistant.\n"
        "For greetings or small talk, reply briefly and warmly without citing sources.\n"
        "For questions, answer using ONLY the context provided and cite sources like [1] or [2].\n"
        "If no context is provided or it doesn't contain the answer, say \"I don't know based on the books.\""
    )

    if results:
        context_parts = []
        for n, (score, p) in enumerate(results, start=1):
            context_parts.append(f"[{n}] ({cite(p)})\n{p['text']}")
        context = "\n\n".join(context_parts)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_prompt = f"Question: {question}\n(No relevant passages were found in the books.)"

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        think=False,
    )
    print("\nAnswer: ", end="")
    for part in response:
        print(part.message.content, end="", flush=True)
    print()

    # Sources only if we actually used any
    if results:
        print("\nSources:")
        for n, (score, p) in enumerate(results, start=1):
            print(f"  [{n}] {cite(p)}  (score {score:.3f})")


def route(question):
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": question},
        ],
        think=False,
        options={"temperature": 0},
    )
    reply = resp.message.content.lower()   # "Nutrition" -> "nutrition" so it still matches
    for r in ROUTES:
        if r in reply:
            return r
    return "general"
        

if __name__ == "__main__":
    client = get_qdrant()

    # Fill the database only if it's empty (ingest is a one-time job)
    if client.get_collection(COLLECTION).points_count == 0:
        print("Reading and chunking PDFs...")
        chunks = chunk_pages(load_pdfs())
        print(f"Embedding and storing {len(chunks):,} chunks in batches of {BATCH_SIZE}...")
        ingest(client, chunks)

    print(f"Ready: {client.get_collection(COLLECTION).points_count:,} chunks in Qdrant.")
    print("Ask a question (press Enter on an empty line to quit).")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            break
        route_name = route(question)                    # ask the receptionist first
        print(f"(route: {route_name})")

        if route_name == "smalltalk":
                results = []                                # no search needed
        else:
                topic = None if route_name == "general" else route_name
                results = search(client, question, top_k=TOP_K, topic=topic)
                results = [(score, p) for score, p in results if score >= MIN_SCORE]

        answer(question, results)

    client.close()
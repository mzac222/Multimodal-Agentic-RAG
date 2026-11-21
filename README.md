# Multimodal Agentic RAG

A local-first AI assistant that answers questions from your own notes **and images**, using a team of agents that plan, research, check and write answers with sources. It runs fully on your own GPU with Ollama.


---

## Scope

### Goals

- Answer questions from personal documents and images, always citing sources
- Run **100% locally** (8B chat model, 8k context)
- Use multiple agents that split work, check each other and decide which tools to use
- **Measure** every retrieval upgrade with an evaluation set instead of guessing
- Offer a free cloud version for a live demo

### In scope

The project is built in six stages, each adding to the one before.

| Stage | What it adds | Key pieces |
|---|---|---|
| **1. RAG from scratch** | End-to-end retrieval and answering with no frameworks | Heading-based chunking, Qwen3 embeddings, cosine similarity (numpy), JSON index, sourced answers |
| **2. Vector database** | Scaling to a larger, real dataset | Qdrant (local mode), PDF ingestion with page + chapter metadata, batched embedding, book › chapter › page citations, topic filters, LLM router (smalltalk / workout / nutrition / general), relevance threshold |
| **3. Better retrieval + evaluation** | Proving which search methods actually help | Test set (30-50 questions), BM25 keyword search, hybrid search, reranker, RAGAS metrics, threshold tuning |
| **4. Multimodal** | Searching images, screenshots and scanned pages | Vision model for image descriptions, mixed text + image results with sources, incremental re-indexing (changed files only) |
| **5. Agents + orchestration** | A team of agents coordinated by a graph | LangGraph, planner / researcher / critic / writer agents, tool use, conversation memory, human approval step, Langfuse observability |
| **6. Cloud deployment** | A free public demo | Groq API for the chat model, Oracle Cloud Always Free server, Qdrant Cloud, Streamlit UI, Langfuse Cloud |

---

## What works today (Stage 2)

- **2 books, 1,423 pages → 1,715 chunks** (300 words each, 50-word overlap), embedded and stored in Qdrant in about a minute on a local GPU
- **Citations down to the page**, e.g. `Human Nutrition (2020 Edition) > Chapter 6. Protein > p.452`
- **Router:** each message is first classified as smalltalk, workout, nutrition or general, so greetings skip search and questions only search the relevant book
- **Grounded answers:** the model answers only from retrieved passages and says *"I don't know based on the books"* when they don't contain the answer

### Known limitations

- The relevance threshold (0.52) is a first guess and is too strict for some workout questions. It gets tuned properly in Stage 3
- Small-model slips: occasional uncited claims or misread numbers in answers
- No conversation memory yet, so each question is answered on its own (Stage 5)
- Incremental re-indexing is deferred: to rebuild after changing books, delete `qdrant_data/` and run again

---

## Data sources

The PDFs are **not included** in this repository. Download them yourself and place them in `docs/pdfs/`.

| Book | Topic | Licence | Link |
|---|---|---|---|
| **FM 7-22 Holistic Health and Fitness** (US Army, 2020) | Workout / fitness | US Government work. Approved for public release; distribution is unlimited | [PDF](https://media.defense.gov/2023/Jul/27/2003268908/-1/-1/1/H2F-FM-7-22-2020.PDF) |
| **Human Nutrition, 2020 Edition** (University of Hawaiʻi at Mānoa Food Science and Human Nutrition Program) | Nutrition | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Some images carry their own licences, credited in the book | [Book page](https://collection.bccampus.ca/textbooks/human-nutrition-2020-edition-university-of-hawaii-at-manoa-155) |

Stage 1 uses two small hand-written sample notes (`docs/gym_notes.md`, `docs/cooking_notes.md`).

> **Disclaimer:** answers are general information drawn from these books, not medical advice.

---

## Tech stack

| Area | Local mode | Cloud mode |
|---|---|---|
| Chat model | Qwen3 8B via Ollama | Groq API |
| Embeddings | `qwen3-embedding:0.6b` via Ollama | Same model, on CPU |
| Vector store | JSON index (Stage 1), then Qdrant (local mode) | Qdrant Cloud (free tier) |
| PDF reading | pypdf | pypdf |
| Orchestration | LangGraph | LangGraph |
| Observability | Langfuse | Langfuse Cloud (Hobby plan) |
| UI | Terminal, then Streamlit | Streamlit |
| Language | Python | Python |

---

## Quick start (Stage 2)

```powershell
# 1. Install
py -m pip install ollama pypdf qdrant-client numpy

# 2. Pull the models
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:8b

# 3. Put the two PDFs in docs/pdfs/, then run
py rag_v2.py
```

The first run reads, chunks and embeds the books. Later runs start straight away.

---

## Roadmap

- [x] Stage 1: RAG from scratch
- [x] Stage 2: Vector database
- [ ] Stage 3: Hybrid retrieval, reranking and evaluation
- [ ] Stage 4: Multimodal (images)
- [ ] Stage 5: Multi-agent orchestration with LangGraph
- [ ] Stage 6: Cloud deployment
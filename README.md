Scope of this Project : 


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
| **2. Vector database** | Scaling to a larger, real dataset | Vector DB, incremental re-indexing (changed files only), metadata filters |
| **3. Better retrieval + evaluation** | Proving which search methods actually help | Test set (30-50 questions), BM25 keyword search, hybrid search, reranker, RAGAS metrics |
| **4. Multimodal** | Searching images, screenshots and scanned pages | Vision model for image descriptions, mixed text + image results with sources |
| **5. Agents + orchestration** | A team of agents coordinated by a graph | LangGraph, planner / researcher / critic / writer agents, tool use, conversation memory, human approval step, Langfuse observability |
| **6. Cloud deployment** | A free public demo | Groq API for the chat model, Oracle Cloud Always Free server, Streamlit UI, Langfuse Cloud |

---

## Tech stack

| Area | Local mode | Cloud mode |
|---|---|---|
| Chat model | Qwen3 8B via Ollama | Groq API |
| Embeddings | `qwen3-embedding:0.6b` via Ollama | Same model, on CPU |
| Vector store | JSON index (Stage 1), then a vector DB | Same vector DB |
| Orchestration | LangGraph | LangGraph |
| Observability | Langfuse | Langfuse Cloud (Hobby plan) |
| UI | Terminal, then Streamlit | Streamlit |
| Language | Python | Python |

---

## Roadmap

- [ ] Stage 1: RAG from scratch
- [ ] Stage 2: Vector database
- [ ] Stage 3: Hybrid retrieval, reranking and evaluation
- [ ] Stage 4: Multimodal (images)
- [ ] Stage 5: Multi-agent orchestration with LangGraph
- [ ] Stage 6: Cloud deployment

# NKP Demo Full RAG — Architecture

## Overview

This app implements a **complete RAG pipeline**:

1. **Chunk** — Split documents into overlapping segments
2. **Embed** — Generate vectors via Ollama (`nomic-embed-text`)
3. **Store** — Save chunks + vectors in Weaviate (vectorizer: none)
4. **Retrieve** — Vector similarity search on user query
5. **Generate** — Build prompt with chunks, call Ollama LLM, return answer

## NKP Catalog Integration

- **Weaviate** and **Ollama** are separate catalog apps, installed and managed by NKP
- This app **connects** to them via Kubernetes DNS—it does not deploy them
- Default URLs assume standard NKP catalog naming:
  - Weaviate: `weaviate.weaviate.svc.cluster.local:80`
  - Ollama: `ollama.ollama.svc.cluster.local:11434`

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Startup / First Request                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  ensure_collection()
       │
       ├── Create Weaviate schema (RAGDocs, vectorizer: none) if needed
       │
       ├── If collection is empty, seed from preload_docs/*.txt:
       │     ├── Chunk text (400 chars, 50 overlap)
       │     ├── Embed each chunk via Ollama /api/embed
       │     └── POST /v1/objects with vector
       │
       └── User can add docs via dropdown (demo_docs) or upload (plain text)

┌─────────────────────────────────────────────────────────────────────────────┐
│  User Query (e.g. "What is Weaviate?")                                       │
└─────────────────────────────────────────────────────────────────────────────┘

  1. Embed query via Ollama
  2. Weaviate nearVector search → top 3 chunks
  3. Build prompt: system + context (chunks) + question
  4. Ollama /api/chat → generated answer
  5. Return answer + source citations
```

## Health Check

`/health` checks both Weaviate and Ollama. Returns 503 if either is unreachable.

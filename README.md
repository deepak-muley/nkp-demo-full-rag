# NKP Demo Full RAG

A **complete RAG application** for the NKP AI catalog. Uses **Weaviate** (vector DB) and **Ollama** (embeddings + LLM) for chunking, semantic search, and answer generation.

## Prerequisites

**Weaviate** and **Ollama** must be enabled in your NKP workspace **before** enabling this app. This app connects to them via in-cluster DNS—it does not install them.

## Catalog Dependency Flow

1. Enable **Weaviate** from the NKP AI catalog
2. Enable **Ollama** from the NKP AI catalog
3. Pull required models in Ollama: `nomic-embed-text` (embeddings), `llama3.2` (or your preferred LLM)
4. Enable **Demo Full RAG** from the NKP AI catalog
5. Open the app URL and ask questions

## Features

- **Chunking** — Documents split into overlapping chunks for better retrieval
- **Embeddings** — Ollama `nomic-embed-text` generates vectors
- **Vector search** — Weaviate stores vectors (vectorizer: none) and performs similarity search
- **LLM generation** — Ollama generates answers grounded in retrieved chunks
- **Source citations** — Shows which documents informed the answer

## Configuration

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `WEAVIATE_URL` | `http://weaviate.weaviate.svc.cluster.local:80` | Weaviate REST API URL |
| `OLLAMA_URL` | `http://ollama-ollama.ollama.svc.cluster.local:11434` | Ollama API URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Chat/LLM model name |

Override via Helm values if your Weaviate or Ollama instances use different namespaces or URLs.

## Required Ollama Models

Before using the app, ensure these models are pulled in Ollama:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Or use smaller alternatives: `llama3.2:1b`, `phi3`, etc. Set `OLLAMA_LLM_MODEL` accordingly.

## Development

### Build and run locally

```bash
pip install -r src/requirements.txt
export WEAVIATE_URL=http://localhost:8080
export OLLAMA_URL=http://localhost:11434
python src/app.py
```

### Build container and Helm chart

```bash
make build
make release VERSION=1.0.0
```

### CI/CD (GitHub Actions)

Same pattern as nkp-demo-rag:

| Trigger | Docker Image | Helm Chart |
|---------|--------------|------------|
| PR to `main` | `ghcr.io/deepak-muley/demo-full-rag:0.0.0-pr.<PR_NUMBER>` | OCI chart with PR version |
| Tag push `v*` | `ghcr.io/deepak-muley/demo-full-rag:<VERSION>` and `:latest` | OCI chart with tag version |

**Release:**

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Catalog Entry (NKP AI Applications Catalog)

When adding to the NKP AI catalog:

- **Catalog path:** `applications/demo-full-rag/1.0.0/`
- **Dependencies:** Weaviate, Ollama (enable first)
- **Source:** [github.com/deepak-muley/nkp-demo-full-rag](https://github.com/deepak-muley/nkp-demo-full-rag)

## License

MIT
# nk-demo-full-rag

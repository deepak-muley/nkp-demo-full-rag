# NKP AI Catalog Entry

When adding this app to the NKP AI Applications Catalog, use the following structure.

## Prerequisites (enable first)

- **Weaviate** — Vector database
- **Ollama** — Embeddings and LLM

## Catalog Directory Structure

```
applications/demo-full-rag/
  1.0.0/
    chart.yaml      # or values override
    metadata.yaml   # app metadata for catalog UI
```

## Helm Chart Reference

- **OCI:** `oci://ghcr.io/deepak-muley/charts/demo-full-rag`
- **Version:** 1.0.0 (or latest from releases)

## Default Configuration

The chart connects to Weaviate and Ollama via in-cluster DNS. Override if your instances use different namespaces:

```yaml
env:
  - name: WEAVIATE_URL
    value: "http://weaviate.weaviate.svc.cluster.local:80"
  - name: OLLAMA_URL
    value: "http://ollama-ollama.ollama.svc.cluster.local:11434"
```

## Model Requirements

Users must pull these models in Ollama before the app works:

- `nomic-embed-text` (embeddings)
- `llama3.2` (or set `OLLAMA_LLM_MODEL` to another model)

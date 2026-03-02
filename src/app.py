#!/usr/bin/env python3
"""
NKP Demo Full RAG — complete RAG application for NKP AI catalog.
Depends on Weaviate (vector DB) and Ollama (embeddings + LLM).
Chunks documents, embeds via Ollama, stores in Weaviate, retrieves by vector search, generates answers via Ollama.
"""
import json
import os
import re
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

WEAVIATE_URL = os.environ.get(
    "WEAVIATE_URL", "http://weaviate.weaviate.svc.cluster.local:80"
).rstrip("/")
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://ollama-ollama.ollama.svc.cluster.local:11434"
).rstrip("/")
COLLECTION_NAME = "RAGDocs"
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>NKP Full RAG</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 1rem; }
    .status { padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .ok { background: #d4edda; color: #155724; }
    .err { background: #f8d7da; color: #721c24; }
    .info { background: #cce5ff; color: #004085; }
    h1 { color: #333; }
    h2 { font-size: 1.1rem; margin-top: 1.5rem; color: #555; }
    input, button { padding: 0.5rem 1rem; font-size: 1rem; }
    .section { margin: 1.5rem 0; padding: 1rem; background: #f8f9fa; border-radius: 8px; }
    .results { margin-top: 1rem; padding: 1rem; background: #fff; border-radius: 8px; border: 1px solid #dee2e6; }
    .answer { margin: 1rem 0; padding: 1rem; background: #e8f4f8; border-radius: 8px; white-space: pre-wrap; }
    .sources { font-size: 0.9em; color: #666; margin-top: 0.5rem; }
    .source { margin: 0.3rem 0; padding-left: 0.5rem; border-left: 3px solid #0d6efd; }
    .upload-form { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
    .upload-form input[type="file"] { flex: 1; min-width: 200px; }
  </style>
</head>
<body>
  <h1>NKP Full RAG</h1>
  <p>Complete RAG: <strong>Weaviate</strong> (vector DB) + <strong>Ollama</strong> (embeddings + LLM). Ask questions and get AI-generated answers grounded in the knowledge base.</p>
  <div class="status {{ status_class }}">{{ message }}</div>

  <div class="section">
    <h2>Ask a question</h2>
    <form method="post" action="/">
      <input type="hidden" name="action" value="query">
      <input type="text" name="query" placeholder="e.g. What is Weaviate? How does RAG work?" value="{{ query }}" style="width: 70%;">
      <button type="submit">Ask</button>
    </form>
  </div>

  <div class="section">
    <h2>Add document from demo library</h2>
    <form method="post" action="/">
      <input type="hidden" name="action" value="add_demo">
      <div class="upload-form">
        <select name="demo_doc" required style="padding: 0.5rem 1rem; font-size: 1rem; min-width: 220px;">
          <option value="">-- Select a document --</option>
          {% for doc in demo_docs %}
          <option value="{{ doc }}">{{ doc }}</option>
          {% endfor %}
        </select>
        <button type="submit">Add to Index</button>
      </div>
    </form>
    {% if add_demo_success %}
    <p class="ok" style="margin-top: 0.5rem; padding: 0.5rem;">✓ Indexed {{ chunks_indexed }} chunk(s) from {{ add_demo_filename }}</p>
    {% endif %}
  </div>

  <div class="section">
    <h2>Upload a document (plain text)</h2>
    <form method="post" action="/" enctype="multipart/form-data">
      <input type="hidden" name="action" value="upload">
      <div class="upload-form">
        <input type="file" name="file" accept=".txt,text/plain" required>
        <button type="submit">Upload & Index</button>
      </div>
    </form>
    {% if upload_success %}
    <p class="ok" style="margin-top: 0.5rem; padding: 0.5rem;">✓ Indexed {{ chunks_indexed }} chunk(s) from {{ upload_filename }}</p>
    {% endif %}
  </div>

  {% if answer and sources %}
  <div class="results">
    <h3>Answer</h3>
    <div class="answer">{{ answer }}</div>
    <div class="sources">
      <strong>Sources:</strong>
      {% for s in sources %}
      <div class="source"><strong>{{ s.title }}</strong>: {{ s.snippet }}</div>
      {% endfor %}
    </div>
  </div>
  {% elif empty_results %}
  <div class="results">
    <p class="info">No relevant documents found. Upload a document above, or try a different question.</p>
  </div>
  {% elif query and error %}
  <div class="results">
    <p class="err">{{ error }}</p>
  </div>
  {% endif %}
</body>
</html>
"""


def weaviate_get(path):
    r = requests.get(f"{WEAVIATE_URL}{path}", timeout=15)
    r.raise_for_status()
    return r.json() if r.content else {}


def weaviate_post(path, json_data=None):
    r = requests.post(f"{WEAVIATE_URL}{path}", json=json_data or {}, timeout=15)
    r.raise_for_status()
    return r.json() if r.content else {}


def ollama_embed(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Ollama. texts can be single string or list."""
    payload = {"model": EMBED_MODEL, "input": texts if len(texts) > 1 else texts[0]}
    r = requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    embeds = data.get("embeddings", [])
    return embeds if isinstance(embeds[0], list) else [embeds]


def ollama_chat(messages: list[dict], stream: bool = False):
    """Generate chat completion from Ollama."""
    payload = {"model": LLM_MODEL, "messages": messages, "stream": stream}
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=120,
        stream=stream,
    )
    r.raise_for_status()
    if stream:
        return r.iter_lines()
    return r.json()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def collection_count() -> int:
    """Return number of objects in the collection."""
    gql = f"""
    {{
      Aggregate {{
        {COLLECTION_NAME} {{
          meta {{
            count
          }}
        }}
      }}
    }}
    """
    try:
        r = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": gql},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        agg = data.get("data", {}).get("Aggregate", {}).get(COLLECTION_NAME, [])
        return agg[0].get("meta", {}).get("count", 0) if agg else 0
    except Exception:
        return 0


def ensure_collection():
    """Create collection if needed, seed from sample_docs when empty."""
    try:
        schema = weaviate_get("/v1/schema")
        exists = any(c.get("class") == COLLECTION_NAME for c in schema.get("classes", []))
    except Exception:
        exists = False

    if not exists:
        weaviate_post(
            "/v1/schema",
            {
                "class": COLLECTION_NAME,
                "vectorizer": "none",
                "properties": [
                    {"name": "title", "dataType": ["text"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "source", "dataType": ["string"]},
                ],
            },
        )

    # Seed from preload_docs when collection is empty
    if collection_count() == 0:
        docs_dir = Path(__file__).parent / "preload_docs"
        if docs_dir.exists():
            for f in docs_dir.glob("*.txt"):
                content = f.read_text()
                title = f.stem.replace("-", " ").title()
                source = f.name
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    try:
                        vectors = ollama_embed([chunk])
                        vec = vectors[0]
                    except Exception:
                        continue
                    obj = {
                        "class": COLLECTION_NAME,
                        "properties": {
                            "title": f"{title} (chunk {i+1})",
                            "content": chunk,
                            "source": source,
                        },
                        "vector": vec,
                    }
                    weaviate_post("/v1/objects", obj)


def vector_search(query: str, limit: int = 3) -> list[dict]:
    """Search Weaviate by vector similarity."""
    try:
        vectors = ollama_embed([query])
        vec = vectors[0]
    except Exception as e:
        raise RuntimeError(f"Ollama embedding failed: {e}") from e

    gql = f"""
    {{
      Get {{
        {COLLECTION_NAME} (
          nearVector: {{ vector: {json.dumps(vec)} }}
          limit: {limit}
        ) {{
          title
          content
          source
          _additional {{ certainty }}
        }}
      }}
    }}
    """
    r = requests.post(
        f"{WEAVIATE_URL}/v1/graphql",
        json={"query": gql},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    items = data.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
    return items


def list_demo_docs() -> list[str]:
    """Return sorted list of .txt filenames in demo_docs."""
    docs_dir = Path(__file__).parent / "demo_docs"
    if not docs_dir.exists():
        return []
    return sorted(f.name for f in docs_dir.glob("*.txt"))


def index_document(content: str, filename: str) -> int:
    """Chunk, embed, and store uploaded document. Returns number of chunks indexed."""
    title = Path(filename).stem.replace("-", " ").replace("_", " ").title()
    chunks = chunk_text(content)
    count = 0
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            vectors = ollama_embed([chunk])
            vec = vectors[0]
        except Exception:
            continue
        obj = {
            "class": COLLECTION_NAME,
            "properties": {
                "title": f"{title} (chunk {i+1})",
                "content": chunk,
                "source": filename,
            },
            "vector": vec,
        }
        weaviate_post("/v1/objects", obj)
        count += 1
    return count


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Build RAG prompt and generate answer via Ollama."""
    context = "\n\n---\n\n".join(
        f"[{c.get('title', '')}]\n{c.get('content', '')}" for c in chunks
    )
    system = """You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context does not contain enough information, say so. Be concise."""
    user = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

    try:
        resp = ollama_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=False,
        )
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}") from e


@app.route("/", methods=["GET", "POST"])
def index():
    """Single page: query + upload + demo dropdown. All docs go to RAGDocs collection."""
    query = ""
    answer = ""
    sources = []
    error = ""
    empty_results = False
    upload_success = False
    upload_filename = ""
    add_demo_success = False
    add_demo_filename = ""
    chunks_indexed = 0
    status_class = "ok"
    message = "Weaviate & Ollama: Connected"
    demo_docs = list_demo_docs()

    try:
        ensure_collection()

        if request.method == "POST":
            action = request.form.get("action", "")
            if action == "upload":
                f = request.files.get("file")
                if f and f.filename:
                    content = f.read().decode("utf-8", errors="replace")
                    chunks_indexed = index_document(content, f.filename)
                    upload_success = True
                    upload_filename = f.filename
            elif action == "add_demo":
                demo_doc = request.form.get("demo_doc", "").strip()
                if demo_doc and demo_doc in demo_docs:
                    docs_dir = Path(__file__).parent / "demo_docs"
                    filepath = docs_dir / demo_doc
                    if filepath.exists():
                        content = filepath.read_text()
                        chunks_indexed = index_document(content, demo_doc)
                        add_demo_success = True
                        add_demo_filename = demo_doc
            elif action == "query":
                query = request.form.get("query", "").strip()
                if query:
                    chunks = vector_search(query, limit=3)
                    if chunks:
                        answer = generate_answer(query, chunks)
                        sources = [
                            {
                                "title": c.get("title", ""),
                                "snippet": (c.get("content", "")[:150] + "...")
                                if len(c.get("content", "")) > 150
                                else c.get("content", ""),
                            }
                            for c in chunks
                        ]
                    else:
                        empty_results = True
    except Exception as e:
        status_class = "err"
        message = f"Error: {e}"
        error = str(e)

    return render_template_string(
        HTML,
        status_class=status_class,
        message=message,
        query=query,
        answer=answer,
        sources=sources,
        error=error,
        empty_results=empty_results,
        upload_success=upload_success,
        upload_filename=upload_filename,
        add_demo_success=add_demo_success,
        add_demo_filename=add_demo_filename,
        chunks_indexed=chunks_indexed,
        demo_docs=demo_docs,
    )


@app.route("/health")
def health():
    status = {}
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/.well-known/ready", timeout=5)
        status["weaviate"] = "ok" if r.status_code == 200 else "unreachable"
    except Exception:
        status["weaviate"] = "unreachable"

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        status["ollama"] = "ok" if r.status_code == 200 else "unreachable"
    except Exception:
        status["ollama"] = "unreachable"

    ok = all(v == "ok" for v in status.values())
    return jsonify(status), 200 if ok else 503


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

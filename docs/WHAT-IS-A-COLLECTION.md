# What is a Collection? (RAG and Vector Databases)

## Overview

In RAG (Retrieval-Augmented Generation) and vector databases, a **collection** is a named container that holds your indexed documents. Think of it like a table in a database or a folder of related content. All the chunks from your documents, along with their vector embeddings, live inside a collection.

## Why Collections Matter

When you ask a question, the RAG system searches within a specific collection. It embeds your question into a vector, then finds the most similar vectors in that collection. The collection defines the scope of the knowledge base: what documents are searchable for each query.

In this app, we use a single collection called `RAGDocs`. Every chunk from sample documents and every uploaded file goes into `RAGDocs`. When you ask "What is AHV?", the system searches `RAGDocs` for the most relevant chunks.

## Collection vs. Document vs. Chunk

- **Document**: A single file or piece of content (e.g. a .txt file about AHV).
- **Chunk**: A small segment of a document. Long documents are split into chunks (e.g. 400 characters each) so they can be embedded and retrieved efficiently.
- **Collection**: The container that holds all chunks (from many documents) and their vectors. One collection can hold chunks from hundreds of documents.

## How It Works in Weaviate

Weaviate is a vector database. Each collection (Weaviate calls it a "class") has:

- **Schema**: Which fields each chunk has (e.g. title, content, source)
- **Stored vectors**: One embedding per chunk, used for similarity search
- **Properties**: The actual text and metadata (title, content, source)

When you upload a document, the app chunks it, embeds each chunk with Ollama, and stores each chunk as one object in the collection. When you query, the app embeds your question, searches the collection for the nearest vectors, and returns the matching chunks to the LLM.

## Collections in Other Systems

The idea is similar across tools:

| System   | Term       |
|----------|------------|
| Weaviate | class      |
| Pinecone | index      |
| Chroma   | collection |
| Qdrant   | collection |

All of these are containers for vectors and metadata that you search over during retrieval.

# rag_search_engine — Keyword / Semantic / Multimodal + Hybrid (RRF) Search, RAG & Evaluation

This repository is a CLI toolkit that can run **keyword search (BM25 / TF-IDF)**, **semantic search (embedding-based)**, and **multimodal search (image → CLIP embedding)** on a movie database; it can also combine them in a **hybrid** way (Weighted / RRF) and optionally apply **LLM-based query enhancement + reranking + evaluation**.

> Note: The repo follows a “traditional/clean” setup: artifacts such as `.env`, `.venv`, caches/embedding outputs, and datasets are not meant to be committed to Git.

---

## Table of Contents
- [0) Motivation](#0-motivation)
- [1) Quick Start](#1-quick-start)
- [2) Data and Cache Logic](#2-data-and-cache-logic)
- [3) Mathematical Background](#3-mathematical-background)
- [4) CLI Commands](#4-cli-commands)
- [5) File-by-File Overview](#5-file-by-file-overview)
- [6) Common Issues](#6-common-issues)

---
## 0) Motivation
This project is a comprehensive search engine toolkit built to demonstrate and deepen my understanding of modern information retrieval systems. It moves beyond basic keyword matching to explore semantic, multimodal, and hybrid search techniques—core components of today's AI-powered applications.

I designed this repository to:

Showcase practical implementation skills across the full search stack: from traditional algorithms (TF-IDF, BM25) to neural embedding models (Sentence Transformers, CLIP) and advanced fusion methods (Reciprocal Rank Fusion).
Solidify conceptual learning by building each component from the ground up. Implementing algorithms like BM25 and RRF provided deep, hands-on insight into their mathematical foundations and trade-offs.
Bridge the gap between retrieval and generation by integrating search with a Large Language Model (Gemini) for query enhancement, reranking, result evaluation, and Retrieval-Augmented Generation (RAG). This creates a complete pipeline from user query to enriched, cited response.
Embrace real-world engineering concerns, such as chunking strategies for long documents, caching embeddings for performance, configurable hybrid scoring, and a modular CLI architecture for extensibility.
Ultimately, this toolkit is a tangible record of my exploration into how search systems think. It reflects a commitment to not just using APIs, but understanding the principles that make them effective, preparing me to build and optimize intelligent search solutions in a professional environment.

## 1) Quick Start

### Installation

- (Recommended) Create an isolated virtual environment
- `python -m venv .venv`
- `source .venv/bin/activate`

- Install the dependencies

---

## 2) Data and Cache Logic
- Data: Movie records are loaded (id, title, description).
- Index/Embedding generation
- Query

---

## 3) Mathematical Background
- TF / IDF / TF-IDF / BM25 / Cosine Similarity / Score Normalization / Weighted Hybrid / Reciprocal Rank Fusion (RRF)

---

## 4) CLI Commands

- Keyword Search CLI
  - `python cli/keyword_search_cli.py build`
  - `python cli/keyword_search_cli.py search "space adventure"`
  - `python cli/keyword_search_cli.py bm25search "space adventure"`
  - `python cli/keyword_search_cli.py tf 12 spaceship`
  - `python cli/keyword_search_cli.py idf spaceship`
  - `python cli/keyword_search_cli.py tfidf 12 spaceship`
  - `python cli/keyword_search_cli.py bm25idf spaceship`
  - `python cli/keyword_search_cli.py bm25tf 12 spaceship`

- Semantic Search CLI
  - `python cli/semantic_search_cli.py verify`
  - `python cli/semantic_search_cli.py search "dream within a dream" --limit 5`
  - `python cli/semantic_search_cli.py embedquery "dream within a dream"`
  - `python cli/semantic_search_cli.py embed_chunks`
  - `python cli/semantic_search_cli.py search_chunked "dream within a dream" --limit 5`

- Multimodal Search CLI
  - `python cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg`
  - `python cli/multimodal_search_cli.py image_search data/paddington.jpeg --limit 5`

- Hybrid Search CLI
  - Weighted
    - `python cli/hybrid_search_cli.py weighted-search "neo the one" --alpha 0.5 --limit 5`

  - RRF
    - `python cli/hybrid_search_cli.py rrf-search "neo the one" --k 60 --limit 5`

- Query enhancement (Gemini)
  - `python cli/hybrid_search_cli.py rrf-search "neo the one" --enhance rewrite --limit 5`

- Rerank options
  - `python cli/hybrid_search_cli.py rrf-search "neo the one" --rerank-method individual --limit 5`
  - `python cli/hybrid_search_cli.py rrf-search "neo the one" --rerank-method batch --limit 5`
  - `python cli/hybrid_search_cli.py rrf-search "neo the one" --rerank-method cross_encoder --limit 5`

- Final result evaluation with LLM (0–3)
  - `python cli/hybrid_search_cli.py rrf-search "neo the one" --evaluate --limit 5`

- RAG / Summarize / Citations / Question (Gemini)
  - `python cli/augmented_generation_cli.py get_rag --query "What should I watch if I like cozy bear movies?" --limit 5`
  - `python cli/augmented_generation_cli.py summarize --query "best sci-fi classics" --limit 5`
  - `python cli/augmented_generation_cli.py citations --query "movies about time travel" --limit 5`
  - `python cli/augmented_generation_cli.py question --query "which movie has X plot?" --limit 5`

- Evaluation (precision/recall/f1 @k)
  - `python cli/evaluation_cli.py --limit 5`

- Query Rewrite with an Image (Gemini)
  - `python cli/describe_image_cli.py --image data/paddington.jpeg --query "family friendly movie"`

- Gemini Connection Test
  - `python cli/gemini_test_cli.py`

---

## 5) File-by-File Overview

- `cli/lib/search_utils.py`: Shared constants, paths, data loading, and a standardized “result format”.
- `cli/lib/keyword_search.py`: Builds an inverted index, tokenizes text, computes TF/IDF/BM25/TF-IDF, and provides keyword search commands.
- `cli/lib/semantic_search.py`: Semantic search using SentenceTransformer embeddings, plus chunk-based search.
- `cli/lib/multimodal_search.py`: `cli/lib/multimodal_search.py`
- `cli/lib/hybrid_search.py`: Normalizes and combines keyword + semantic results (weighted or RRF).

- `cli/keyword_search_cli.py`: Argparse CLI for `build`, `search`, `tf`, `idf`, `tfidf`, `bm25idf`, `bm25tf`, `bm25search`.
- `cli/semantic_search_cli.py`: Argparse CLI for `verify`, `embed_text`, `verify_embeddings`, `embedquery`, `search`, `chunk`, `semantic_chunk`, `embed_chunks`, `search_chunked`.
- `cli/multimodal_search_cli.py`: Argparse CLI for `verify_image_embedding`, `image_search`.
- `cli/hybrid_search_cli.py`: Argparse CLI for `normalize`, `weighted-search`, `rrf-search` (with `enhance`, `rerank-method`, `evaluate` options).
- `cli/augmented_generation_cli.py`: Argparse CLI for `get_rag`, `summarize`, `citations`, `question`.
- `cli/evaluation_cli.py`: Computes precision / recall@k / f1.
- `cli/describe_image_cli.py`: Describes an image and combines it with a query.
- `cli/gemini_test_cli.py`: Gemini API connection test.

---


# rag_search_engine — Keyword / Semantic / Multimodal + Hybrid (RRF) Search, RAG & Evaluation

This repository is a CLI toolkit that can run **keyword search (BM25 / TF-IDF)**, **semantic search (embedding-based)**, and **multimodal search (image → CLIP embedding)** on a movie database; it can also combine them in a **hybrid** way (Weighted / RRF) and optionally apply **LLM-based query enhancement + reranking + evaluation**.

> Note: The repo follows a "traditional/clean" setup: artifacts such as `.env`, `.venv`, caches/embedding outputs, and datasets are not meant to be committed to Git.

---

## Table of Contents
- [0) Motivation](#0-motivation)
- [1) Quick Start](#1-quick-start)
- [2) Data and Cache Logic](#2-data-and-cache-logic)
- [3) Mathematical Background](#3-mathematical-background)
- [4) CLI Commands](#4-cli-commands)
- [5) Usage Guide](#5-usage-guide)
- [6) File-by-File Overview](#6-file-by-file-overview)
- [7) Common Issues](#7-common-issues)

---
## 0) Motivation

This project is a comprehensive search engine toolkit built to demonstrate and deepen my understanding of modern information retrieval systems. It moves beyond basic keyword matching to explore semantic, multimodal, and hybrid search techniques—core components of today's AI-powered applications.

I designed this repository to:

* **Showcase practical implementation skills** across the full search stack: from traditional algorithms (TF-IDF, BM25) to neural embedding models (Sentence Transformers, CLIP) and advanced fusion methods (Reciprocal Rank Fusion).
* **Solidify conceptual learning** by building each component from the ground up. Implementing algorithms like BM25 and RRF provided deep, hands-on insight into their mathematical foundations and trade-offs.
* **Bridge the gap between retrieval and generation** by integrating search with a Large Language Model (Gemini) for query enhancement, reranking, result evaluation, and Retrieval-Augmented Generation (RAG). This creates a complete pipeline from user query to enriched, cited response.
* **Embrace real-world engineering concerns**, such as chunking strategies for long documents, caching embeddings for performance, configurable hybrid scoring, and a modular CLI architecture for extensibility.

Ultimately, this toolkit is a tangible record of my exploration into how search systems *think*. It reflects a commitment to not just using APIs, but understanding the principles that make them effective, preparing me to build and optimize intelligent search solutions in a professional environment.

## 1) Quick Start

### Installation

- (Recommended) Create an isolated virtual environment
- `python -m venv .venv`
- `source .venv/bin/activate`

- Install the dependencies
- `pip install -r requirements.txt`

---

## 2) Data and Cache Logic

- **Data**: Movie records are loaded from `data/movies.json` (id, title, description).
- **Index Generation**: Keyword indexes and embeddings are generated on first run and cached locally.
- **Query Processing**: All search queries go through configurable pipelines with optional caching.

---

## 3) Mathematical Background

- **TF / IDF / TF-IDF**: Traditional term weighting schemes
- **BM25**: Probabilistic relevance scoring with term saturation
- **Cosine Similarity**: Vector similarity measure for embeddings
- **Score Normalization**: Min-max scaling for score combination
- **Weighted Hybrid**: Linear combination of normalized scores
- **Reciprocal Rank Fusion (RRF)**: Rank-based fusion without score normalization

---

## 4) CLI Commands

### Keyword Search CLI
```bash
python cli/keyword_search_cli.py build
python cli/keyword_search_cli.py search "space adventure"
python cli/keyword_search_cli.py bm25search "space adventure"
python cli/keyword_search_cli.py tf 12 spaceship
python cli/keyword_search_cli.py idf spaceship
python cli/keyword_search_cli.py tfidf 12 spaceship
python cli/keyword_search_cli.py bm25idf spaceship
python cli/keyword_search_cli.py bm25tf 12 spaceship
Semantic Search CLI
```

```bash
python cli/semantic_search_cli.py verify
python cli/semantic_search_cli.py search "dream within a dream" --limit 5
python cli/semantic_search_cli.py embedquery "dream within a dream"
python cli/semantic_search_cli.py embed_chunks
python cli/semantic_search_cli.py search_chunked "dream within a dream" --limit 5
Multimodal Search CLI
````

```bash
python cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
python cli/multimodal_search_cli.py image_search data/paddington.jpeg --limit 5
Hybrid Search CLI
```
```bash
# Weighted Hybrid
python cli/hybrid_search_cli.py weighted-search "neo the one" --alpha 0.5 --limit 5
```


# RRF Hybrid
```bash
python cli/hybrid_search_cli.py rrf-search "neo the one" --k 60 --limit 5
```


# With Query Enhancement
```bash
python cli/hybrid_search_cli.py rrf-search "neo the one" --enhance rewrite --limit 5
```


# With Reranking
```bash
python cli/hybrid_search_cli.py rrf-search "neo the one" --rerank-method individual --limit 5
python cli/hybrid_search_cli.py rrf-search "neo the one" --rerank-method batch --limit 5
python cli/hybrid_search_cli.py rrf-search "neo the one" --rerank-method cross_encoder --limit 5
```
# With LLM Evaluation
```bash
python cli/hybrid_search_cli.py rrf-search "neo the one" --evaluate --limit 5
```
RAG / Generation CLI

```bash
python cli/augmented_generation_cli.py get_rag --query "What should I watch if I like cozy bear movies?" --limit 5
python cli/augmented_generation_cli.py summarize --query "best sci-fi classics" --limit 5
python cli/augmented_generation_cli.py citations --query "movies about time travel" --limit 5
python cli/augmented_generation_cli.py question --query "which movie has X plot?" --limit 5
```
```bash
# Evaluation & Utility CLI
python cli/evaluation_cli.py --limit 5
python cli/describe_image_cli.py --image data/paddington.jpeg --query "family friendly movie"
python cli/gemini_test_cli.py
```
# 5) Usage Guide

This toolkit offers multiple search methods, each accessible via dedicated CLI commands. Below are detailed explanations of when and how to use each component.

## 5.1 Keyword Search (Traditional IR)

When to use it:

Testing specific term frequency (TF) calculations
Understanding inverse document frequency (IDF) in your dataset
Comparing BM25 vs TF-IDF scoring
When search terms are very specific and unambiguous
Examples:

```bash
# Build the inverted index first
python cli/keyword_search_cli.py build
```

# Search using TF-IDF scoring
```bash
python cli/keyword_search_cli.py search "space adventure"
```

# Search using BM25 scoring
```bash
python cli/keyword_search_cli.py bm25search "space adventure"
```
5.2 Semantic Search (Embedding-Based)

When to use it:

Searching with paraphrased queries
Finding content with similar meaning but different wording
When users describe concepts rather than use specific terms
Working with long-form text descriptions
Examples:

```bash
# Standard semantic search on movie descriptions
python cli/semantic_search_cli.py search "dream within a dream" --limit 5
```
```bash
# Chunk-based search (for long documents)
python cli/semantic_search_cli.py embed_chunks
python cli/semantic_search_cli.py search_chunked "dream within a dream" --limit 5
```
## 5.3 Multimodal Search (Image-to-Text)

When to use it:

Finding movies based on visual style or scenes
When users have visual references but no text descriptions
Cross-modal retrieval applications
Examples:

```bash
python cli/multimodal_search_cli.py image_search data/paddington.jpeg --limit 5
```
## 5.4 Hybrid Search (Combining Methods)

When to use it:

Production scenarios where recall and precision both matter
When queries contain both specific terms and conceptual meaning
Balancing between exact matching and semantic understanding
Weighted Hybrid:

```bash
# Adjust alpha (0-1) to balance keyword vs semantic
python cli/hybrid_search_cli.py weighted-search "neo the one" --alpha 0.5 --limit 5
Reciprocal Rank Fusion (RRF):
```
```bash
# RRF combines rankings without score normalization issues
python cli/hybrid_search_cli.py rrf-search "neo the one" --k 60 --limit 5
```


## 5.5 Advanced Features with LLM Integration

Query Enhancement:

```bash
python cli/hybrid_search_cli.py rrf-search "neo the one" --enhance rewrite --limit 5
Reranking Options:

bash
python cli/hybrid_search_cli.py rrf-search "neo the one" --rerank-method individual --limit 5
LLM-Based Evaluation:

bash
python cli/hybrid_search_cli.py rrf-search "neo the one" --evaluate --limit 5
```
## 5.6 RAG (Retrieval-Augmented Generation)

When to use it:

Answering complex questions about movies
Generating summaries based on multiple results
Creating cited responses for research or analysis
Examples:

```bash
python cli/augmented_generation_cli.py get_rag --query "What should I watch if I like cozy bear movies?" --limit 5
python cli/augmented_generation_cli.py summarize --query "best sci-fi classics" --limit 5
```
## 5.7 Workflow Examples

Basic Search Pipeline:

```bash
# 1. Build keyword index
python cli/keyword_search_cli.py build

# 2. Generate embeddings (first time)
python cli/semantic_search_cli.py embed_chunks

# 3. Run hybrid search with enhancement
python cli/hybrid_search_cli.py rrf-search "your query" --enhance rewrite --limit 10

# 4. Evaluate results
python cli/hybrid_search_cli.py rrf-search "your query" --evaluate --limit 10
RAG Pipeline:

bash
# 1. Retrieve relevant documents
python cli/hybrid_search_cli.py rrf-search "movie about artificial intelligence" --limit 5

# 2. Generate comprehensive answer with citations
python cli/augmented_generation_cli.py get_rag --query "movie about artificial intelligence" --limit 5
```
## 5.8 Tips for Effective Usage

Start Simple: Begin with basic keyword or semantic search before using hybrid methods
Cache Wisely: The system caches embeddings - first run will be slower
Limit Results: Use --limit parameter to control output, especially during testing
Experiment with Alpha: In weighted hybrid, try different alpha values (0.1, 0.3, 0.5, 0.7, 0.9)
Use Evaluation: Run the evaluation CLI to quantitatively compare different search configurations
Check API Limits: Be mindful of Gemini API usage when using enhancement, reranking, or evaluation features

# 6) File-by-File Overview

Core Library Modules

cli/lib/search_utils.py: Shared constants, paths, data loading, and standardized "result format"
cli/lib/keyword_search.py: Builds inverted index, tokenizes text, computes TF/IDF/BM25/TF-IDF
cli/lib/semantic_search.py: Semantic search using SentenceTransformer embeddings, plus chunk-based search
cli/lib/multimodal_search.py: Image embedding generation and multimodal search using CLIP
cli/lib/hybrid_search.py: Normalizes and combines keyword + semantic results (weighted or RRF)
CLI Entry Points

cli/keyword_search_cli.py: Argparse CLI for build, search, tf, idf, tfidf, bm25idf, bm25tf, bm25search
cli/semantic_search_cli.py: Argparse CLI for verify, embed_text, verify_embeddings, embedquery, search, chunk, semantic_chunk, embed_chunks, search_chunked
cli/multimodal_search_cli.py: Argparse CLI for verify_image_embedding, image_search
cli/hybrid_search_cli.py: Argparse CLI for normalize, weighted-search, rrf-search (with enhance, rerank-method, evaluate options)
cli/augmented_generation_cli.py: Argparse CLI for get_rag, summarize, citations, question
cli/evaluation_cli.py: Computes precision / recall@k / f1
cli/describe_image_cli.py: Describes an image and combines it with a query
cli/gemini_test_cli.py: Gemini API connection test
Data & Configuration

data/movies.json: Movie dataset with IDs, titles, and descriptions
requirements.txt: Python dependencies
.env.example: Template for environment variables (API keys)
# 7) Common Issues

Installation Issues

ModuleNotFoundError: Ensure you've activated the virtual environment and run pip install -r requirements.txt
CUDA errors: If using GPU, ensure PyTorch with CUDA support is installed
Runtime Issues

Slow first run: Embedding generation happens on first use and is cached for subsequent runs
API key errors: Set GEMINI_API_KEY in your .env file for LLM features
Memory issues: Reduce batch size in embedding generation or use --limit to reduce result sets
Search Quality Issues

Poor keyword results: Try BM25 instead of TF-IDF (bm25search instead of search)
Poor semantic results: Try query enhancement (--enhance rewrite)
Hybrid not working well: Adjust alpha parameter or switch to RRF (rrf-search)
Performance Tips

Use chunked search for long documents
Enable caching for repeated queries
Use batch reranking for large result sets
Consider disabling LLM features for faster, lighter searches


# RAG-Powered Domain-Specific Q&A System — Complete User Guide

## H9DLGA: Deep Learning and Generative AI — Spring 2026, National College of Ireland

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [The Research Question](#2-the-research-question)
3. [Key Concepts & Intuition](#3-key-concepts--intuition)
4. [System Architecture](#4-system-architecture)
5. [Project Structure](#5-project-structure)
6. [Setup & Installation](#6-setup--installation)
7. [Module-by-Module Deep Dive](#7-module-by-module-deep-dive)
8. [The Document Corpus](#8-the-document-corpus)
9. [Notebook Workflow](#9-notebook-workflow)
10. [Experiments & What They Test](#10-experiments--what-they-test)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Configuration Reference](#12-configuration-reference)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What This Project Is

This is a **research system** that builds a Retrieval-Augmented Generation (RAG) pipeline for answering questions about **Irish Climate Action Policy**, then rigorously compares it against a baseline approach (asking an LLM the same questions without any document retrieval).

The system:
- Ingests 14 Irish government climate policy documents (~300K words total)
- Splits them into searchable chunks using multiple strategies
- Embeds chunks into vector space using multiple embedding models
- Stores embeddings in a ChromaDB vector database
- At query time, retrieves the most relevant chunks and feeds them as context to an LLM
- Evaluates answers against 50 manually curated ground-truth Q&A pairs
- Compares RAG answers vs. baseline LLM answers across 7 metrics

---

## 2. The Research Question

> **How does Retrieval-Augmented Generation (RAG) compare to pure LLM prompting for domain-specific question answering in terms of faithfulness, answer relevancy, and factual accuracy?**

### Why This Matters

Large Language Models (LLMs) like LLaMA 3.1 have broad general knowledge, but they:
- **Hallucinate**: They confidently produce incorrect facts, especially for niche domains
- **Lack currency**: Their training data has a cutoff date — they don't know about recent Irish policy changes
- **Can't cite sources**: You can't verify where their answer came from

RAG solves these problems by giving the LLM **actual documents to read** before answering. The hypothesis is that this grounding in real source material produces more faithful, accurate, and verifiable answers.

---

## 3. Key Concepts & Intuition

### 3.1 What is RAG?

**Retrieval-Augmented Generation** is a two-stage process:

```
Stage 1: RETRIEVAL — Find relevant document passages
Stage 2: GENERATION — Feed those passages + the question to an LLM
```

Think of it like an open-book exam vs. a closed-book exam:
- **Baseline (closed-book)**: The LLM answers from memory alone
- **RAG (open-book)**: The LLM can look up relevant passages before answering

### 3.2 Why Chunking?

Documents are too long to feed entirely to an LLM (context window limits + noise). So we **chunk** them — split them into smaller pieces (typically 500-1000 characters). At query time, we only retrieve the 3-5 most relevant chunks, keeping the context focused and within token limits.

Different chunking strategies exist because **where you cut matters**:
- **Fixed-size**: Simple character-count splits. Fast but may cut mid-sentence.
- **Recursive**: Tries to split at paragraph breaks first, then sentences, then words. Preserves semantic meaning.
- **Sentence-based**: Splits only at sentence boundaries. Ensures grammatical completeness.

### 3.3 Why Embeddings?

To find "relevant" chunks for a query, we need a way to measure meaning-similarity between text. **Embeddings** convert text into numerical vectors (arrays of numbers) such that semantically similar texts have vectors pointing in similar directions.

For example:
- "Ireland's 2030 emissions target" and "51% reduction by 2030" → high similarity
- "Ireland's 2030 emissions target" and "Paris Agreement signatories" → lower similarity

We use **cosine similarity** to measure how closely two embedding vectors align (1.0 = identical meaning, 0.0 = unrelated).

### 3.4 Why a Vector Store?

With 2,000+ chunks, we need an efficient way to find the top-K most similar chunks to a query. A **vector store** (ChromaDB in this project) indexes all chunk embeddings so that similarity search is fast — no need to compare the query against every single chunk linearly.

### 3.5 Why Multiple Experiments?

There is no single "best" configuration for RAG. Performance depends on:
- How you chunk (strategy, size, overlap)
- Which embedding model you use
- How many chunks you retrieve (top-K)
- Which LLM generates the answer

This project systematically varies each parameter to find the best combination and to understand the sensitivity of each choice.

---

## 4. System Architecture

### 4.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE / INDEXING PHASE                      │
│                                                                  │
│   Raw Documents (PDFs, TXT)                                      │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                                │
│   │ data_loader  │  Extract text, clean, normalize               │
│   └──────┬──────┘                                                │
│          │                                                        │
│          ▼                                                        │
│   ┌─────────────┐                                                │
│   │   chunker    │  Split into 500-1000 char chunks              │
│   └──────┬──────┘                                                │
│          │                                                        │
│          ▼                                                        │
│   ┌─────────────┐     ┌──────────────┐                           │
│   │  embeddings  │────▶│ vector_store  │  ChromaDB on disk       │
│   └─────────────┘     └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    ONLINE / QUERY PHASE                           │
│                                                                  │
│   User Question                                                  │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                                │
│   │  embeddings  │  Embed the query                              │
│   └──────┬──────┘                                                │
│          │                                                        │
│          ▼                                                        │
│   ┌──────────────┐                                               │
│   │ vector_store  │  Find top-K similar chunks                   │
│   └──────┬───────┘                                               │
│          │                                                        │
│          ▼                                                        │
│   ┌──────────────┐                                               │
│   │ rag_pipeline  │  Format context + question into prompt       │
│   └──────┬───────┘                                               │
│          │                                                        │
│          ▼                                                        │
│   ┌──────────────┐                                               │
│   │  llm_client   │  Send to Groq/Ollama, get answer            │
│   └──────┬───────┘                                               │
│          │                                                        │
│          ▼                                                        │
│   ┌──────────────┐                                               │
│   │  evaluator    │  Score against ground truth                  │
│   └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Dependency Graph

```
config.py ◄──────── Used by ALL modules (paths, keys, parameters)
    │
    ├── data_loader.py ──► chunker.py ──► vector_store.py
    │                                          │
    │                    embeddings.py ◄───────┤
    │                         │                │
    │                         ▼                ▼
    │                    rag_pipeline.py ◄── llm_client.py
    │                         │
    │                         ▼
    │                    evaluator.py
    │
    └── All notebooks import from src/
```

### 4.3 Two Execution Modes

| Mode | How It Works | Purpose |
|------|-------------|---------|
| **RAG Mode** | Retrieve chunks → Build context → LLM answers with context | Test grounded generation |
| **Baseline Mode** | LLM answers from its own knowledge (no retrieval) | Control group for comparison |

---

## 5. Project Structure

```
dl/
├── data/
│   ├── raw/                          # 14 original source documents
│   │   ├── CCAC-AR2026-2-*.pdf       # Climate advisory council report
│   │   ├── Energy-in-Ireland-2025.pdf # SEAI energy statistics
│   │   ├── climate-action-plan-*.pdf  # Government's main climate strategy
│   │   ├── carbon_capture_and_storage.txt
│   │   ├── climate_change_in_europe.txt
│   │   ├── climate_change_mitigation.txt
│   │   ├── electricity_sector_in_ireland.txt
│   │   ├── emissions_trading.txt
│   │   ├── energy_policy_of_the_european_union.txt
│   │   ├── environmental_protection_agency_ireland.txt
│   │   ├── european_green_deal.txt
│   │   ├── paris_agreement.txt
│   │   ├── sustainable_energy_authority_of_ireland.txt
│   │   └── transport_in_ireland.txt
│   ├── processed/                     # Cleaned .txt versions + metadata JSON
│   │   ├── doc_00_CCAC-*.txt ... doc_14_transport_in_ireland.txt
│   │   └── documents_metadata.json
│   └── ground_truth/
│       └── qa_pairs.json              # 50 curated question-answer pairs
├── src/                               # Core Python modules
│   ├── config.py                      # Central configuration
│   ├── data_loader.py                 # Document extraction & cleaning
│   ├── chunker.py                     # Text chunking strategies
│   ├── embeddings.py                  # Embedding model wrapper
│   ├── vector_store.py                # ChromaDB interface
│   ├── llm_client.py                  # Groq + Ollama LLM client
│   ├── rag_pipeline.py                # End-to-end RAG orchestration
│   └── evaluator.py                   # Evaluation metrics
├── notebooks/                         # Jupyter workflow (run in order)
│   ├── 01_data_collection.ipynb
│   ├── 02_chunking_analysis.ipynb
│   ├── 03_embedding_comparison.ipynb
│   ├── 04_rag_pipeline.ipynb
│   ├── 05_baseline_llm.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_results_visualization.ipynb
├── results/
│   ├── metrics/                       # Experiment results (JSON)
│   │   ├── chunking_experiment_results.json
│   │   ├── embedding_comparison_results.json
│   │   ├── rag_batch_results.json
│   │   ├── baseline_model_comparison.json
│   │   └── eval_main_evaluation.json
│   └── figures/                       # Generated visualization charts
├── vector_store/                      # ChromaDB persistent storage
├── groq_test.py                       # Quick test for Groq API connectivity
├── requirements.txt                   # Python dependencies
├── .env                               # Your environment config (not in git)
├── .env.example                       # Template for .env
└── README.md                          # Project overview
```

---

## 6. Setup & Installation

### 6.1 Prerequisites

- Python 3.10+ 
- pip (Python package manager)
- ~2 GB disk space (for models and vector store)
- Internet access (for Groq API and downloading embedding models)

### 6.2 Step-by-Step Setup

```bash
# 1. Navigate to project
cd /home/umukesh/Desktop/dl

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings (see below)
```

### 6.3 Choosing Your LLM Provider

You have two options:

#### Option A: Groq (Cloud — Recommended for Quick Start)

Groq provides free API access to open-source LLMs with very fast inference.

1. Go to https://console.groq.com and create a free account
2. Generate an API key
3. Set in your `.env` file:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```
4. Test the connection:
   ```bash
   python groq_test.py
   ```

**Rate limits**: Free tier allows ~30 requests/minute. The system includes a 2.5-second delay between requests to stay within limits.

#### Option B: Ollama (Local — No API Key Needed)

Ollama runs LLMs on your own machine. Requires more RAM (~8 GB for 8B models).

1. Install Ollama from https://ollama.ai
2. Pull the model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
4. Set in your `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.1:8b
   OLLAMA_BASE_URL=http://localhost:11434
   ```

### 6.4 Running the Notebooks

```bash
jupyter notebook notebooks/
```

Run them **in numerical order** (01 through 07). Each notebook builds on the outputs of the previous one.

---

## 7. Module-by-Module Deep Dive

### 7.1 `src/config.py` — Central Configuration

**Rationale**: Every module needs paths, API keys, and parameter values. Rather than scattering these across files, `config.py` loads everything from a `.env` file and exposes it as importable Python constants.

**What it does**:
- Loads `.env` using `python-dotenv`
- Defines all directory paths and creates them if missing
- Configures LLM provider (Groq vs Ollama) with model names
- Maps embedding model nicknames to HuggingFace model IDs
- Defines chunking experiment configurations
- Contains prompt templates for RAG and baseline modes
- Sets up logging

**Key design decision**: Environment-variable-based configuration means you can switch between Groq and Ollama, or change the embedding model, by editing `.env` — no code changes needed.

**Helper functions**:
- `get_active_llm_config()` — Returns a dict with the active provider's settings
- `get_embedding_model_id(name)` — Resolves "bge" → "BAAI/bge-small-en-v1.5"
- `print_config()` — Prints current configuration for debugging

---

### 7.2 `src/data_loader.py` — Document Loading & Preprocessing

**Rationale**: Raw documents come in different formats (PDF, plain text) and contain noise (encoding artifacts, page numbers, excessive whitespace). This module normalizes everything into clean text with metadata.

**What it does**:

1. **Loads PDFs** using PyPDF2 — extracts text page by page
2. **Loads text files** with UTF-8 encoding
3. **Cleans text** by:
   - Removing encoding artifacts (replacement characters)
   - Normalizing whitespace (tabs → spaces, collapsing multiple spaces)
   - Removing standalone page numbers
   - Collapsing excessive newlines (3+ → 2)
4. **Attaches metadata** to each document (source file, type, word count, title, description)
5. **Saves processed versions** as clean `.txt` files + a `documents_metadata.json`

**Data flow**:
```
data/raw/*.pdf, *.txt
    → load_document() → extract + clean
    → save_processed_documents()
    → data/processed/doc_XX_name.txt + documents_metadata.json
```

**Usage in notebooks**: Called in `01_data_collection.ipynb`

---

### 7.3 `src/chunker.py` — Text Chunking Strategies

**Rationale**: An LLM's context window is limited, and feeding an entire 112K-word document would be wasteful and noisy. Chunking breaks documents into focused passages that can be individually embedded and retrieved.

**The `Chunk` dataclass**:
```python
@dataclass
class Chunk:
    text: str              # The chunk content
    chunk_id: int          # Sequential ID within document
    source_doc: str        # Which document it came from
    start_char: int        # Character position in original
    end_char: int          # Character position in original
    strategy: str          # "fixed", "recursive", or "sentence"
    chunk_size_config: int # Configured max size
    overlap_config: int    # Configured overlap
```

**Three strategies implemented**:

| Strategy | How It Works | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| **Fixed** | Split every N characters | Simple, predictable chunk count | May cut mid-sentence or mid-word |
| **Recursive** | Try `\n\n` first, then `\n`, then `.`, then ` `, then character | Preserves semantic boundaries at multiple levels | Slightly more complex |
| **Sentence** | Split on sentence boundaries (`.!?`), group until size limit | Always grammatically complete | May produce uneven chunk sizes |

**Overlap**: Each chunk overlaps with the previous one by N characters. This ensures that information near chunk boundaries isn't lost. For example, with 200-char overlap, the last 200 characters of chunk #5 also appear as the first 200 characters of chunk #6.

**Why recursive is the default**: It gives the best balance — it tries to split at paragraph breaks (preserving topical coherence), falls back to sentence breaks, and only splits mid-sentence as a last resort.

**Experiment configurations tested**:
| Strategy | Chunk Size | Overlap | Resulting Chunks |
|----------|-----------|---------|-----------------|
| Fixed | 512 | 50 | ~3,320 |
| Fixed | 1024 | 100 | ~1,665 |
| Recursive | 500 | 50 | ~4,131 |
| Recursive | 1000 | 200 | ~2,096 |
| Sentence | 1000 | 200 | ~1,803 |

---

### 7.4 `src/embeddings.py` — Embedding Model Wrapper

**Rationale**: We want to compare multiple embedding models without changing any downstream code. This module provides a unified `EmbeddingModel` class that abstracts away the differences.

**Three models supported**:

| Name | Model ID | Dimensions | Backend | Characteristics |
|------|----------|-----------|---------|----------------|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | sentence-transformers | Fast, lightweight, good general purpose |
| `bge` | `BAAI/bge-small-en-v1.5` | 384 | sentence-transformers | Better retrieval quality, uses query prefix |
| `nomic` | `nomic-ai/nomic-embed-text-v1.5` | 768 | Ollama | Higher dimensional, requires Ollama running |

**BGE query prefix**: The BGE model was trained with a special prefix for queries. When embedding a query (not a document chunk), the model prepends:
```
"Represent this sentence for searching relevant passages: {query}"
```
This asymmetric approach improves retrieval accuracy.

**Key methods**:
- `embed_texts(texts)` — Batch-embed a list of strings (for indexing documents)
- `embed_query(query)` — Embed a single query (with prefix for BGE)
- `compute_similarity(a, b)` — Cosine similarity between two embeddings

---

### 7.5 `src/vector_store.py` — ChromaDB Vector Store

**Rationale**: Once chunks are embedded, we need a database that can efficiently find the most similar embeddings to a query. ChromaDB provides persistent, on-disk vector storage with built-in similarity search.

**Why ChromaDB**: It's lightweight (no separate server needed), stores data on disk (survives restarts), and integrates easily with Python. For a research project of this scale (~2,000 vectors), it's ideal.

**How it works**:

1. **Indexing** (`add_chunks`):
   - Takes a list of `Chunk` objects and an embedding model
   - Generates embeddings for all chunks in batches
   - Stores each chunk's text, embedding, and metadata in a ChromaDB collection
   - Metadata includes: source document, chunk ID, strategy, word count

2. **Searching** (`search`):
   - Takes a query string and embedding model
   - Embeds the query
   - Finds the top-K most similar chunks using cosine distance
   - Converts distances to similarity scores: `similarity = 1 - distance`
   - Filters out results below the similarity threshold
   - Returns ranked results with scores and metadata

**Collection naming**: Each experiment can use its own collection (e.g., `"rag_bge_recursive_1000"`) to avoid mixing results.

**Persistence**: Data is stored in `vector_store/` on disk. Once indexed, you don't need to re-embed — just reopen the collection.

---

### 7.6 `src/llm_client.py` — LLM Interface

**Rationale**: The project supports two LLM providers with different APIs. This module provides a unified interface so the RAG pipeline doesn't care which provider is being used.

**Architecture**:
```
BaseLLMClient (abstract)
    ├── GroqClient    — Cloud API (Groq)
    └── OllamaClient  — Local inference (Ollama)
```

**GroqClient**:
- Uses the `groq` Python SDK
- Rate-limited to ~30 requests/minute (free tier)
- Built-in 2.5-second sleep between requests to avoid hitting limits
- Returns token counts for analysis

**OllamaClient**:
- Calls the local Ollama HTTP API
- No rate limits (limited by your hardware speed)
- Requires the model to be pulled and Ollama to be running

**Two prompt modes**:

**RAG mode** — The LLM is instructed to ONLY use the provided context:
```
System: You are a helpful assistant... You must ONLY use the provided context...
User:   Context: [retrieved chunks] | Question: [user question] | Answer based ONLY on context:
```

**Baseline mode** — The LLM answers from its own knowledge:
```
System: You are a helpful assistant... Provide answers based on your knowledge.
User:   Question: [user question] | Answer:
```

**Factory function**: `get_llm_client(provider)` returns the right client based on configuration.

**Response structure**:
```python
{
    "response": "Ireland targets 51% reduction...",
    "model": "llama-3.1-8b-instant",
    "provider": "groq",
    "latency_seconds": 2.34,
    "input_tokens": 523,
    "output_tokens": 87,
    "mode": "rag"
}
```

---

### 7.7 `src/rag_pipeline.py` — RAG Orchestration

**Rationale**: This is the central orchestrator that ties together retrieval, context building, and generation into a single pipeline. It's the module you interact with to ask questions.

**The `RAGPipeline` class** connects three components:
```python
pipeline = RAGPipeline(
    vector_store=store,         # Where chunks are stored
    embedding_model=emb_model,  # How to embed queries
    llm_client=llm_client       # Which LLM to use
)
```

**Query flow** (the `answer()` method):

```
1. retrieve(query)
   └── Embed query → Search vector store → Get top-K chunks

2. build_context(chunks)
   └── Format chunks into a single string:
       "[Source 1: ireland_climate_action_plan.txt (relevance: 0.87)]
        Ireland has a legally binding target...
        ---
        [Source 2: CCAC_report.txt (relevance: 0.82)]
        The Climate Action Plan sets out..."

3. generate_rag_response(question, context)
   └── Send formatted prompt to LLM → Get answer

4. Return result dict with answer, chunks, timings, token counts
```

**Baseline comparison** (the `answer_baseline()` method): Sends the question to the LLM without any retrieval or context.

**Batch processing** (the `batch_answer()` method): Processes a list of questions, optionally generating both RAG and baseline answers for each.

---

### 7.8 `src/evaluator.py` — Evaluation Metrics

**Rationale**: To scientifically compare RAG vs. baseline, we need quantitative metrics. This module implements 5 metrics that measure different aspects of answer quality.

**Metrics explained**:

#### ROUGE-L (Textual Overlap)
- Measures the **longest common subsequence** between the generated answer and ground truth
- Captures structural similarity — do the answers share the same phrasing?
- Returns precision, recall, and F1 score
- Example: Ground truth "51% reduction by 2030" vs answer "Ireland aims for 51% reduction by 2030" → high ROUGE-L

#### Answer Similarity (Semantic Match)
- Embeds both the generated answer and ground truth, measures cosine similarity
- Captures **meaning** rather than exact wording — paraphrases score well
- Falls back to Jaccard similarity if no embedding model is available
- Range: 0.0 (completely different meaning) to 1.0 (identical meaning)

#### Faithfulness (Grounding in Context)
- Measures what fraction of **content words** in the answer also appear in the retrieved context
- Filters out 130+ English stop words (the, is, at, which, etc.)
- High faithfulness = the answer sticks to what the documents say
- Low faithfulness = the answer includes information not in the context (possible hallucination)
- **Only applicable to RAG mode** (baseline has no context)

#### Context Precision (Retrieval Quality)
- Of the chunks retrieved, what fraction are actually **relevant** to the ground truth answer?
- A chunk is "relevant" if it shares >10% word overlap with the ground truth
- High precision = the retriever found on-topic chunks
- Low precision = the retriever brought back irrelevant noise

#### Context Recall (Retrieval Completeness)
- What fraction of the **key terms** in the ground truth were found somewhere in the retrieved chunks?
- High recall = all the information needed to answer was retrieved
- Low recall = some needed information was missed by retrieval

**Evaluation workflow**:
```python
# Evaluate one question
result = evaluate_single(
    question="What is Ireland's 2030 target?",
    rag_answer="Ireland targets 51% reduction...",
    baseline_answer="Ireland aims to reduce...",
    ground_truth="51% reduction by 2030...",
    context="[Retrieved chunks text]",
    retrieved_chunks=[...],
    embedding_model=emb_model
)

# Evaluate all 50 questions at once
batch_eval = evaluate_batch(all_results, ground_truths, emb_model)

# Get aggregate metrics with improvement percentages
aggregate = compute_aggregate_metrics(batch_eval)
```

**Improvement calculation**: `(RAG_metric - Baseline_metric) / Baseline_metric * 100%`

---

## 8. The Document Corpus

### 8.1 Source Documents (14 total)

| # | Document | Words | Description |
|---|----------|-------|-------------|
| 1 | CCAC Climate Report 2026 | ~11,500 | Climate Change Advisory Council annual review |
| 2 | Energy in Ireland 2025 | ~60,900 | Comprehensive SEAI energy statistics report |
| 3 | Carbon Capture & Storage | ~6,950 | CCS technology overview |
| 4 | Ireland Climate Action Plan 2023 | ~112,550 | **Primary source** — government's full climate strategy |
| 5 | Climate Change in Europe | ~7,750 | European climate impacts analysis |
| 6 | Climate Change Mitigation | ~11,550 | Global mitigation strategies overview |
| 7 | Electricity Sector in Ireland | ~1,170 | Irish electricity infrastructure summary |
| 8 | Emissions Trading | ~5,350 | EU Emissions Trading System explanation |
| 9 | EU Energy Policy | ~4,735 | European Union energy framework |
| 10 | EPA Ireland | ~305 | Environmental Protection Agency role summary |
| 11 | European Green Deal | ~7,520 | EU green transition plan |
| 12 | Paris Agreement | ~5,960 | International climate accord |
| 13 | SEAI | ~350 | Sustainable Energy Authority role summary |
| 14 | Transport in Ireland | ~2,420 | Irish transport sector analysis |

**Total**: ~300,000 words across 14 documents.

### 8.2 Ground Truth Dataset

The file `data/ground_truth/qa_pairs.json` contains **50 manually curated Q&A pairs** covering 10 categories:

| Category | Example Question |
|----------|-----------------|
| Targets | "What is Ireland's 2030 greenhouse gas emissions reduction target?" |
| Carbon Budgets | "What are Ireland's three carbon budgets and their allocations?" |
| Electricity | "What percentage of electricity does Ireland aim to generate from renewables by 2030?" |
| Transport | "How many electric vehicles does Ireland target by 2030?" |
| Buildings | "How many home retrofits is Ireland targeting by 2030?" |
| Agriculture | Questions about agricultural emissions and targets |
| Carbon Tax | Questions about carbon pricing mechanisms |
| EU Policy | Questions about the European Green Deal and EU ETS |
| International | Questions about the Paris Agreement |
| Governance | Questions about CCAC, EPA, SEAI roles |

Each Q&A pair includes the ground truth answer and the source document it came from.

---

## 9. Notebook Workflow

Run the notebooks in sequence. Each builds on the previous notebook's outputs.

### Notebook 01: Data Collection
**Purpose**: Load all 14 raw documents, clean them, save processed versions.

**What happens**:
- Iterates over `data/raw/` files
- Extracts text from PDFs and text files
- Applies cleaning pipeline (artifacts, whitespace, page numbers)
- Saves clean `.txt` files to `data/processed/`
- Generates `documents_metadata.json` with statistics
- Displays corpus statistics (word counts, document sizes)

**Output**: `data/processed/` directory populated with clean text files.

---

### Notebook 02: Chunking Analysis
**Purpose**: Test 5 chunking configurations and analyze their behavior.

**What happens**:
- Loads processed documents
- Runs each chunking experiment configuration:
  - Fixed 512/50, Fixed 1024/100
  - Recursive 500/50, Recursive 1000/200
  - Sentence 1000/200
- For each: counts chunks, measures size distributions
- Compares strategies visually
- Saves results to `results/metrics/chunking_experiment_results.json`

**Key insight**: Recursive 1000/200 gives ~2,096 well-formed chunks — a good balance between granularity and semantic coherence.

---

### Notebook 03: Embedding Comparison
**Purpose**: Compare the three embedding models on sample texts.

**What happens**:
- Loads sample texts and queries
- Embeds them with MiniLM, BGE, and Nomic
- Compares embedding speed
- Computes intra-document and query-document similarities
- Saves results to `results/metrics/embedding_comparison_results.json`

**Key insight**: BGE tends to produce better retrieval-oriented embeddings due to its training methodology and query prefix feature.

---

### Notebook 04: RAG Pipeline
**Purpose**: Build the full RAG pipeline and run it on all 50 ground truth questions.

**What happens**:
- Initializes embedding model, vector store, and LLM client
- Indexes all chunks into ChromaDB (if not already done)
- Runs each of the 50 questions through the RAG pipeline
- Records answers, retrieval times, generation times, token counts
- Saves results to `results/metrics/rag_batch_results.json`

**Output**: RAG answers for all 50 questions with full metadata.

---

### Notebook 05: Baseline LLM
**Purpose**: Get baseline (no-retrieval) answers for comparison.

**What happens**:
- Uses the same LLM but WITHOUT retrieval
- Sends each of the 50 questions directly to the LLM
- Records answers and latencies
- Saves results to `results/metrics/baseline_model_comparison.json`

**Output**: Baseline answers for all 50 questions.

---

### Notebook 06: Evaluation
**Purpose**: Score both RAG and baseline answers against ground truth.

**What happens**:
- Loads RAG results, baseline results, and ground truth Q&A pairs
- For each question, computes all 5 metrics for both RAG and baseline
- Computes aggregate (mean) metrics across all 50 questions
- Calculates improvement percentages
- Saves full evaluation to `results/metrics/eval_main_evaluation.json`

**Output**: Per-question and aggregate evaluation metrics.

---

### Notebook 07: Results Visualization
**Purpose**: Create publication-ready charts and tables.

**What happens**:
- Loads evaluation results
- Generates comparison charts:
  - RAG vs Baseline metric comparison (bar charts)
  - Per-category performance breakdown
  - Chunking strategy comparison charts
  - Embedding model comparison charts
  - Latency analysis
- Saves figures to `results/figures/`

**Output**: Charts and summary tables for the research report.

---

## 10. Experiments & What They Test

### Experiment 1: Chunking Strategy Comparison
**Question**: Does it matter how we split documents?
**Variables**: 5 configurations (fixed/recursive/sentence × different sizes)
**Measured**: Chunk count, size distribution, downstream retrieval quality

### Experiment 2: Embedding Model Comparison
**Question**: Which embedding model produces the best retrieval?
**Variables**: MiniLM vs BGE vs Nomic
**Measured**: Embedding speed, query-document similarity, retrieval precision

### Experiment 3: LLM Comparison
**Question**: Does the LLM choice affect answer quality?
**Variables**: LLaMA 3.1 8B vs Mixtral 8x7B vs Gemma 2 9B (via Groq)
**Measured**: ROUGE-L, answer similarity, faithfulness, latency

### Experiment 4: RAG vs Baseline (The Main Experiment)
**Question**: Does retrieval actually help?
**Variables**: RAG (with context) vs Baseline (without context)
**Measured**: All 7 metrics across 50 questions

### Experiment 5: Retrieval Parameter Sensitivity
**Question**: How many chunks should we retrieve?
**Variables**: top-K = 3, 5, 7, 10 + varying similarity thresholds
**Measured**: Context precision, context recall, answer quality

---

## 11. Evaluation Framework

### 11.1 Metrics at a Glance

| Metric | What It Measures | Range | Higher = Better? |
|--------|-----------------|-------|-----------------|
| ROUGE-L F1 | Text overlap with ground truth | 0–1 | Yes |
| Answer Similarity | Semantic similarity to ground truth | 0–1 | Yes |
| Faithfulness | Grounding in retrieved context | 0–1 | Yes |
| Context Precision | Relevance of retrieved chunks | 0–1 | Yes |
| Context Recall | Completeness of retrieval | 0–1 | Yes |
| Latency | Response time | Seconds | No (lower = better) |
| Token Count | Tokens used (cost proxy) | Count | Neutral |

### 11.2 Why These Metrics?

- **ROUGE-L** is the classic text generation metric — does the answer contain the same information?
- **Answer Similarity** captures meaning beyond exact wording — "51% cut" and "reduction of 51 percent" should match
- **Faithfulness** is unique to RAG — it checks if the LLM stayed faithful to what the documents say vs. hallucinating
- **Context Precision/Recall** evaluate the retriever separately from the generator — you need good retrieval for good answers

### 11.3 How Improvement Is Calculated

```
Improvement % = (RAG_score - Baseline_score) / Baseline_score × 100
```

A positive improvement means RAG outperformed the baseline on that metric.

---

## 12. Configuration Reference

### 12.1 Environment Variables (`.env`)

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq`, `ollama` | Which LLM backend to use |
| `GROQ_API_KEY` | *(empty)* | Your key | API key from console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | See below | Which Groq model to use |
| `OLLAMA_MODEL` | `llama3.1:8b` | Any pulled model | Which Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL | Ollama server address |
| `EMBEDDING_MODEL` | `bge` | `minilm`, `bge`, `nomic` | Embedding model |
| `CHUNKING_STRATEGY` | `recursive` | `fixed`, `recursive`, `sentence` | How to split documents |
| `CHUNK_SIZE` | `1000` | Integer | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Integer | Character overlap between chunks |
| `TOP_K` | `5` | Integer | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.5` | 0.0–1.0 | Minimum similarity to include a chunk |

### 12.2 Available LLM Models

**Via Groq (cloud)**:
| Model | Description |
|-------|-------------|
| `llama-3.1-8b-instant` | Fast, good quality (default) |
| `mixtral-8x7b-32768` | Mixture-of-experts, strong reasoning |
| `gemma2-9b-it` | Google's model, instruction-tuned |

**Via Ollama (local)**:
| Model | Description |
|-------|-------------|
| `llama3.1:8b` | Meta's LLaMA 3.1 |
| `mistral:7b` | Mistral AI's base model |
| `gemma2:9b` | Google's Gemma 2 |

---

## 13. Troubleshooting

### "GROQ_API_KEY not set" or authentication errors
- Ensure your `.env` file has `GROQ_API_KEY=gsk_...` (no quotes around the value)
- Run `python groq_test.py` to test connectivity

### Groq rate limit errors (429)
- The free tier allows ~30 requests/minute
- The system has a built-in 2.5s delay, but large batch runs may still hit limits
- Wait a minute and retry, or reduce batch size

### Ollama connection refused
- Make sure Ollama is running: `ollama serve`
- Check it's on the right port: `curl http://localhost:11434/api/tags`
- Make sure you've pulled the model: `ollama pull llama3.1:8b`

### ChromaDB errors or stale data
- Delete the `vector_store/` directory and re-run notebook 04 to rebuild
- Each experiment should use its own collection name to avoid conflicts

### Embedding model download slow
- First run downloads the model (~100 MB for MiniLM/BGE). This is a one-time cost.
- Models are cached in `~/.cache/huggingface/`

### Out of memory
- Use `minilm` or `bge` (384 dimensions) instead of `nomic` (768 dimensions)
- For Ollama, ensure you have at least 8 GB RAM for 8B models
- Reduce batch size in config: `BATCH_SIZE=5`

### Notebook kernel dies
- Usually a memory issue — restart kernel and run cells sequentially
- Don't run all 7 notebooks in the same kernel session

---

## Quick Reference: Running a Single Question

```python
from src.config import *
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.llm_client import get_llm_client
from src.rag_pipeline import RAGPipeline

# Initialize components
embedding_model = EmbeddingModel("bge")
store = VectorStore("rag_bge_recursive_1000")  # use existing collection
llm_client = get_llm_client("groq")

# Create pipeline
pipeline = RAGPipeline(store, embedding_model, llm_client)

# Ask a question (RAG mode)
result = pipeline.answer("What is Ireland's 2030 emissions target?")
print(result["answer"])
print(f"Retrieved {result['num_chunks_retrieved']} chunks in {result['retrieval_time']:.2f}s")

# Compare with baseline
baseline = pipeline.answer_baseline("What is Ireland's 2030 emissions target?")
print(baseline["answer"])
```

---

*This guide was generated for the H9DLGA Deep Learning and Generative AI project, Spring 2026.*

# RAG-Powered Domain-Specific Q&A System

## H9DLGA: Deep Learning and Generative AI — Project 2026

### Research Question
> *How does Retrieval-Augmented Generation (RAG) compare to pure LLM prompting for domain-specific question answering in terms of faithfulness, answer relevancy, and factual accuracy?*

**Domain:** Irish Climate Action Policy — leveraging publicly available government documents, policy frameworks, and climate reports (e.g., gov.ie, EPA, SEAI).

---

## 1. Explanation, Rationale, and Intuition

### The Problem: LLM Hallucination
Large Language Models (LLMs) like LLaMA-3.1 or GPT-4 possess vast general knowledge but lack specific, up-to-date domain knowledge (like exact 2030 Irish emission targets). If you ask a "Baseline" LLM a specific question, it will confidently guess ("hallucinate") an answer based on statistical likelihoods.

### The Solution: Retrieval-Augmented Generation (RAG)
RAG forces the LLM into an "open-book" exam. 
1. **Intution:** Instead of letting the LLM guess, we first take authentic Irish Government PDFs, chop them into small paragraphs ("chunks"), and store them using **Vector Embeddings** (which converts textual meaning into math arrays).
2. **Rationale:** When a user asks a question, we convert the question into math, heavily search the Vector Database for the most mathematically similar (semantically relevant) chunks, and then give those chunks to the LLM. 
3. **Outcome:** The LLM's only job is to read the extremely accurate paragraphs we just handed it, and answer the question natively. 

### Why CRISP-DM?
This project uses **CRISP-DM** (Cross-Industry Standard Process for Data Mining) heavily expected in academic settings to prove your concept methodically through Jupyter Notebooks, testing different chunk sizes, embedding algorithms (BGE vs MiniLM), and measuring outputs via RAGAS metrics (Faithfulness & Context Precision).

---

## 2. Detailed File Structure

The project was built specifically to separate modular logic (`src/`) from sequential experimentation (`notebooks/`).

```text
dl/
├── data/
│   ├── raw/                     # Place your downloaded raw authentic PDFs here.
│   ├── processed/               # Extracted and cleaned raw text.
│   └── ground_truth/
│       └── qa_pairs.json        # 51 Hand-crafted valid Q&A pairs to mathematically evaluate accuracy against.
│
├── notebooks/                   # The CRISP-DM execution lifecycle (Run these sequentially)
│   ├── 01_data_collection.ipynb    # Loads text and PDFs into standard strings.
│   ├── 02_chunking_analysis.ipynb  # Tests chopping text by fixed size vs. recursive sizes.
│   ├── 03_embedding_comparison.ipynb # Embeds Text into ChromaDB.
│   ├── 04_rag_pipeline.ipynb       # GENERATION: Feeds vectors to Groq/Ollama and generates "rag_answers".
│   ├── 05_baseline_llm.ipynb       # Baselining to compare with generic LLM generation.
│   ├── 06_evaluation.ipynb         # Grades Faithfulness via Semantic Sim & precision logic without API bias. 
│   └── 07_results_visualization.ipynb # Charts radar graphs for the final IEEE report.
│
├── src/                         # Reusable Python logic
│   ├── config.py                # Environment Configuration
│   ├── data_loader.py           # Automates PyPDF2 parsing logic
│   ├── chunker.py               # Houses recursive/sentence chunking algorithms
│   ├── embeddings.py            # Wraps Huggingface SentenceTransformers (BGE/MiniLM)
│   ├── vector_store.py          # Wraps ChromaDB vector interactions
│   ├── rag_pipeline.py          # The core pipeline algorithm tying Retrieval to LLM Generation
│   ├── llm_client.py            # Interfaces automatically handing rate limits between Groq Cloud and Ollama local
│   └── evaluator.py             # Math bounds to calculate ROUGE-L overlapping and Faithfulness
│
├── results/                     # All empirical outputs
│   ├── metrics/                 # JSON dumps containing all generation logs.
│   └── figures/                 # All `.png` visual graphs.
│
├── .env                         # Your active API keys and configuration (Local state)
├── .env.example                 # Example variables to copy over
└── requirements.txt             # Dependency Library
```

---

## 3. Guide to Running the System

### Step 1: Install Dependencies
Ensure you are in the `dl/` directory, create a virtual environment, and install dependencies to avoid overriding global packages.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure Environment (`.env`)
Copy the base configuration over.
```bash
cp .env.example .env
```
Open `.env` and set your preferred LLM Provider (Currently configured with rate-limiting safety).
*   **Groq API (Cloud - Fast)**: `LLM_PROVIDER=groq` and provide your `GROQ_API_KEY`. (Built-in 2.5s inference limits to prevent 429 timeouts on the free tier).
*   **Ollama (Local - Secure)**: `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3.1:8b`. Make sure to pull the model to your system (`ollama pull llama3.1:8b`).

### Step 3: Run the Pipeline via Notebooks
To evaluate the project empirically for your report, start your Jupyter Kernel.
```bash
jupyter notebook notebooks/
```
**Execute the notebooks in sequence `01` to `07`**.
1. `01` through `03` will parse your PDFs and build your local ChromaDB Vector storage locally.
2. `04` and `05` will execute ~100 API queries internally. *Note: Under the Groq API free tier, this will deliberately take 5-7 minutes due to safe rate throttling.*
3. `06` and `07` will crunch the generated text, comparing it against the benchmark `qa_pairs.json`, and output beautiful graphs to your `results/figures/` folder. 

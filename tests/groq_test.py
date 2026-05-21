import json
import time
from pathlib import Path

from src.config import logger
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.llm_client import get_llm_client
from src.rag_pipeline import RAGPipeline
from src.evaluator import evaluate_batch, load_ground_truth, print_evaluation_summary
from src.data_loader import load_all_documents
from src.chunker import chunk_all_documents

# Initialize components
logger.info("Initializing models and vector store...")
embedding_model = EmbeddingModel('bge')

# Check if vector store has chunks, else populate it
store = VectorStore(collection_name='rag_main_bge_recursive')
if store.get_collection_info().get('count', 0) == 0:
    docs = load_all_documents()
    chunks = chunk_all_documents(docs, strategy='recursive', chunk_size=1000, chunk_overlap=200)
    store.add_chunks(chunks, embedding_model)

client = get_llm_client('groq')
pipeline = RAGPipeline(store, embedding_model, client)

# Load GT
gt_path = Path('data/ground_truth/qa_pairs.json')
ground_truth = load_ground_truth(gt_path)

# Run test on 5 items
questions = list(ground_truth.keys())[:5]
print(f"Starting evaluation of {len(questions)} questions using Groq API...")

results = pipeline.batch_answer(questions, top_k=5, include_baseline=True, temperature=0.1)

# Evaluate
print("Evaluating results...")
eval_data = evaluate_batch(results, ground_truth, embedding_model)
print_evaluation_summary(eval_data['aggregate'])

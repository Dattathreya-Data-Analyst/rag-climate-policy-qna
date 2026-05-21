"""
Embedding Models Module for the RAG Q&A System.

Provides a unified interface for multiple embedding models, supporting
sentence-transformers (MiniLM, BGE) and Ollama-based models (Nomic).
Includes utilities for batch embedding and model comparison.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from src.config import (
    EMBEDDING_MODEL, EMBEDDING_MODEL_MAP, EMBEDDING_MODELS_TO_COMPARE,
    OLLAMA_BASE_URL, logger
)


class EmbeddingModel:
    """
    Unified embedding model wrapper supporting multiple backends.
    
    Supports:
    - sentence-transformers models (MiniLM, BGE) via HuggingFace
    - Ollama-hosted models (Nomic) via local API
    
    Usage:
        model = EmbeddingModel("bge")
        embeddings = model.embed_texts(["Hello world", "How are you?"])
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Model identifier ("minilm", "bge", or "nomic").
                       Defaults to EMBEDDING_MODEL from config.
        """
        self.model_name = (model_name or EMBEDDING_MODEL).lower()
        self.model = None
        self.model_id = EMBEDDING_MODEL_MAP.get(self.model_name)
        self.dimensions = None
        self._load_time = None
        
        if not self.model_id:
            raise ValueError(
                f"Unknown embedding model: {self.model_name}. "
                f"Available: {list(EMBEDDING_MODEL_MAP.keys())}"
            )
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model into memory."""
        start_time = time.time()
        
        if self.model_name == "nomic":
            # Nomic model uses Ollama backend
            self._backend = "ollama"
            logger.info(f"Using Ollama backend for {self.model_id}")
            # Test connection
            try:
                import ollama
                self.model = ollama.Client(host=OLLAMA_BASE_URL)
                # Do a test embedding to verify model is available
                test = self.model.embeddings(model="nomic-embed-text", prompt="test")
                self.dimensions = len(test["embedding"])
            except Exception as e:
                logger.warning(
                    f"Ollama not available for nomic model: {e}. "
                    f"Falling back to sentence-transformers."
                )
                self._backend = "sentence_transformers"
                self._load_sentence_transformer()
        else:
            self._backend = "sentence_transformers"
            self._load_sentence_transformer()
        
        self._load_time = time.time() - start_time
        logger.info(
            f"Loaded embedding model '{self.model_name}' "
            f"(dims={self.dimensions}, backend={self._backend}, "
            f"load_time={self._load_time:.2f}s)"
        )
    
    def _load_sentence_transformer(self):
        """Load a sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_id)
        self.dimensions = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (sentence-transformers only)
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if self._backend == "ollama":
            return self._embed_ollama(texts)
        else:
            return self._embed_sentence_transformer(texts, batch_size, show_progress)
    
    def _embed_sentence_transformer(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings)
    
    def _embed_ollama(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Ollama API."""
        embeddings = []
        for text in texts:
            response = self.model.embeddings(
                model="nomic-embed-text",
                prompt=text,
            )
            embeddings.append(response["embedding"])
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        For BGE models, prepends the query instruction prefix for
        better retrieval performance as recommended by the model authors.
        
        Args:
            query: Query string to embed
            
        Returns:
            1D NumPy array of the query embedding
        """
        # BGE models benefit from a query prefix
        if self.model_name == "bge":
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        embedding = self.embed_texts([query], show_progress=False)
        return embedding[0]
    
    def get_info(self) -> Dict:
        """Return model information as a dictionary."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "dimensions": self.dimensions,
            "backend": self._backend,
            "load_time_seconds": self._load_time,
        }


def compute_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    
    Args:
        embedding_a: First embedding vector
        embedding_b: Second embedding vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def compare_embedding_models(
    sample_texts: List[str],
    sample_queries: List[str],
    model_names: List[str] = None,
) -> Dict:
    """
    Compare multiple embedding models on the same dataset.
    
    Evaluates each model on embedding time, dimensions, and
    query-document similarity scores.
    
    Args:
        sample_texts: List of document texts to embed
        sample_queries: List of queries to test retrieval quality
        model_names: List of model identifiers to compare
        
    Returns:
        Dictionary with comparison results for each model
    """
    model_names = model_names or EMBEDDING_MODELS_TO_COMPARE
    results = {}
    
    for model_name in model_names:
        logger.info(f"Evaluating embedding model: {model_name}")
        
        try:
            # Initialize model
            model = EmbeddingModel(model_name)
            
            # Time the embedding process
            start_time = time.time()
            doc_embeddings = model.embed_texts(sample_texts, show_progress=False)
            embed_time = time.time() - start_time
            
            # Test query similarities
            query_results = []
            for query in sample_queries:
                query_embedding = model.embed_query(query)
                similarities = [
                    compute_similarity(query_embedding, doc_emb)
                    for doc_emb in doc_embeddings
                ]
                query_results.append({
                    "query": query,
                    "max_similarity": float(max(similarities)),
                    "mean_similarity": float(np.mean(similarities)),
                    "min_similarity": float(min(similarities)),
                })
            
            results[model_name] = {
                "model_info": model.get_info(),
                "embed_time_seconds": embed_time,
                "docs_per_second": len(sample_texts) / embed_time if embed_time > 0 else 0,
                "query_results": query_results,
                "avg_max_similarity": np.mean([r["max_similarity"] for r in query_results]),
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results

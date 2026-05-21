"""
Vector Store Module for the RAG Q&A System.

Manages ChromaDB collections for storing and retrieving document chunk
embeddings. Supports multiple collections for comparing different 
embedding/chunking configurations.
"""

import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import VECTOR_STORE_DIR, TOP_K, SIMILARITY_THRESHOLD, logger
from src.chunker import Chunk
from src.embeddings import EmbeddingModel


class VectorStore:
    """
    ChromaDB-based vector store for document chunk retrieval.
    
    Manages embedding storage, similarity search, and collection
    lifecycle. Supports creating multiple collections for experiments
    comparing different chunking/embedding strategies.
    
    Usage:
        store = VectorStore(collection_name="rag_bge_recursive")
        store.add_chunks(chunks, embedding_model)
        results = store.search("What is Ireland's climate target?", embedding_model, top_k=5)
    """
    
    def __init__(
        self,
        collection_name: str = "rag_default",
        persist_dir: Path = None,
    ):
        """
        Initialize the vector store with a ChromaDB collection.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_dir: Directory to persist the database (defaults to VECTOR_STORE_DIR)
        """
        self.persist_dir = persist_dir or VECTOR_STORE_DIR
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        
        logger.info(
            f"Vector store initialized: '{collection_name}' "
            f"({self.collection.count()} existing documents)"
        )
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embedding_model: EmbeddingModel,
        batch_size: int = 100,
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Generates embeddings for all chunks and stores them in ChromaDB
        with associated metadata.
        
        Args:
            chunks: List of Chunk objects to store
            embedding_model: EmbeddingModel instance for generating embeddings
            batch_size: Number of chunks to process per batch
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        logger.info(f"Embedding {len(chunks)} chunks with {embedding_model.model_name}...")
        
        # Extract texts for embedding
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        start_time = time.time()
        embeddings = embedding_model.embed_texts(texts)
        embed_time = time.time() - start_time
        logger.info(f"Embedding completed in {embed_time:.2f}s")
        
        # Prepare data for ChromaDB
        ids = [f"{self.collection_name}_chunk_{chunk.chunk_id}_{chunk.source_doc[:20]}" 
               for chunk in chunks]
        metadatas = [
            {
                "source_doc": chunk.source_doc,
                "chunk_id": str(chunk.chunk_id),
                "strategy": chunk.strategy,
                "word_count": str(chunk.word_count),
                "char_count": str(chunk.char_count),
                "start_char": str(chunk.start_char),
                "end_char": str(chunk.end_char),
            }
            for chunk in chunks
        ]
        
        # Add in batches
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            total_added += batch_end - i
        
        logger.info(
            f"Added {total_added} chunks to collection '{self.collection_name}' "
            f"(total: {self.collection.count()})"
        )
        
        return total_added
    
    def search(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        top_k: int = None,
        similarity_threshold: float = None,
    ) -> List[Dict]:
        """
        Search for the most relevant chunks given a query.
        
        Args:
            query: Search query string
            embedding_model: EmbeddingModel instance for generating query embedding
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1) to include results
            
        Returns:
            List of result dictionaries with keys:
              - text: Retrieved chunk text
              - score: Cosine similarity score
              - metadata: Chunk metadata
        """
        top_k = top_k or TOP_K
        similarity_threshold = similarity_threshold or SIMILARITY_THRESHOLD
        
        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        
        # Parse results
        parsed_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                # ChromaDB returns distances (lower = more similar for cosine)
                # Convert distance to similarity score
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Cosine distance to similarity
                
                if similarity >= similarity_threshold:
                    parsed_results.append({
                        "text": results["documents"][0][i],
                        "score": round(similarity, 4),
                        "metadata": results["metadatas"][0][i],
                        "rank": i + 1,
                    })
        
        logger.debug(
            f"Search returned {len(parsed_results)} results "
            f"(query: '{query[:50]}...')"
        )
        
        return parsed_results
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_dir": str(self.persist_dir),
        }
    
    def clear_collection(self):
        """Delete all documents from the current collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared collection '{self.collection_name}'")
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection '{self.collection_name}'")


def create_experiment_store(
    experiment_name: str,
    chunks: List[Chunk],
    embedding_model: EmbeddingModel,
) -> VectorStore:
    """
    Create a new vector store for an experiment configuration.
    
    Convenience function that creates a collection, embeds chunks,
    and returns the ready-to-use vector store.
    
    Args:
        experiment_name: Name for the experiment collection
        chunks: List of Chunk objects to store
        embedding_model: EmbeddingModel instance
        
    Returns:
        Initialized and populated VectorStore
    """
    store = VectorStore(collection_name=experiment_name)
    store.clear_collection()  # Start fresh
    store.add_chunks(chunks, embedding_model)
    return store

"""
RAG Pipeline Module for the RAG Q&A System.

Orchestrates the full RAG (Retrieval-Augmented Generation) pipeline:
query → retrieve → augment → generate. Supports configurable components
and experiment tracking.
"""

import time
import json
from typing import List, Dict, Optional
from pathlib import Path

from src.config import TOP_K, SIMILARITY_THRESHOLD, logger
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.llm_client import (
    BaseLLMClient, get_llm_client,
    generate_rag_response, generate_baseline_response,
)
from src.chunker import Chunk, chunk_all_documents


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    
    Connects the vector store, embedding model, and LLM to answer
    questions using retrieved context from the document corpus.
    
    Usage:
        pipeline = RAGPipeline(
            vector_store=store,
            embedding_model=model,
            llm_client=client,
        )
        result = pipeline.answer("What are Ireland's 2030 climate targets?")
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        llm_client: BaseLLMClient = None,
        top_k: int = None,
        similarity_threshold: float = None,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Initialized and populated VectorStore instance
            embedding_model: EmbeddingModel for query embedding
            llm_client: LLM client for generation (creates default if None)
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score for retrieval
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_client = llm_client or get_llm_client()
        self.top_k = top_k or TOP_K
        self.similarity_threshold = similarity_threshold or SIMILARITY_THRESHOLD
        
        # Query history for tracking/debugging
        self.query_history = []
        
        logger.info(
            f"RAG Pipeline initialized: "
            f"store='{vector_store.collection_name}', "
            f"embedding='{embedding_model.model_name}', "
            f"llm='{self.llm_client.get_info()['model']}', "
            f"top_k={self.top_k}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None,
    ) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return (overrides pipeline default)
            similarity_threshold: Minimum similarity (overrides pipeline default)
            
        Returns:
            List of retrieved chunk dictionaries
        """
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        results = self.vector_store.search(
            query=query,
            embedding_model=self.embedding_model,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        
        return results
    
    def build_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Build a context string from retrieved chunks.
        
        Formats the retrieved chunks into a single context string
        suitable for insertion into the LLM prompt.
        
        Args:
            retrieved_chunks: List of retrieval results from self.retrieve()
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk["metadata"].get("source_doc", "Unknown")
            score = chunk["score"]
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n"
                f"{chunk['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def answer(
        self,
        question: str,
        top_k: int = None,
        temperature: float = 0.1,
        return_context: bool = True,
    ) -> Dict:
        """
        Answer a question using the full RAG pipeline.
        
        Pipeline: query → retrieve → build context → generate answer
        
        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            temperature: LLM temperature for generation
            return_context: Whether to include retrieved context in results
            
        Returns:
            Dictionary with:
              - answer: Generated answer text
              - question: Original question
              - retrieved_chunks: List of retrieved chunks (if return_context)
              - context: Formatted context string (if return_context)
              - retrieval_time: Time for retrieval step
              - generation_time: Time for LLM generation
              - total_time: Total pipeline time
              - model_info: LLM and embedding model information
        """
        total_start = time.time()
        
        # Step 1: Retrieve relevant chunks
        retrieval_start = time.time()
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Build context from retrieved chunks
        context = self.build_context(retrieved_chunks)
        
        # Step 3: Generate answer using LLM with context
        generation_start = time.time()
        llm_result = generate_rag_response(
            question=question,
            context=context,
            client=self.llm_client,
            temperature=temperature,
        )
        generation_time = time.time() - generation_start
        
        total_time = time.time() - total_start
        
        # Compile result
        result = {
            "answer": llm_result["response"],
            "question": question,
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3),
            "num_chunks_retrieved": len(retrieved_chunks),
            "llm_model": llm_result["model"],
            "llm_provider": llm_result["provider"],
            "embedding_model": self.embedding_model.model_name,
            "top_k": top_k or self.top_k,
            "input_tokens": llm_result.get("input_tokens", 0),
            "output_tokens": llm_result.get("output_tokens", 0),
        }
        
        if return_context:
            result["retrieved_chunks"] = retrieved_chunks
            result["context"] = context
        
        # Track in history
        self.query_history.append({
            "question": question,
            "answer": result["answer"],
            "total_time": total_time,
            "num_chunks": len(retrieved_chunks),
        })
        
        return result
    
    def answer_baseline(
        self,
        question: str,
        temperature: float = 0.1,
    ) -> Dict:
        """
        Answer a question using the baseline (no retrieval) approach.
        
        Sends the question directly to the LLM without any context.
        Used as a comparison baseline for evaluating RAG effectiveness.
        
        Args:
            question: The question to answer
            temperature: LLM temperature for generation
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        llm_result = generate_baseline_response(
            question=question,
            client=self.llm_client,
            temperature=temperature,
        )
        
        total_time = time.time() - start_time
        
        return {
            "answer": llm_result["response"],
            "question": question,
            "total_time": round(total_time, 3),
            "llm_model": llm_result["model"],
            "llm_provider": llm_result["provider"],
            "mode": "baseline",
            "input_tokens": llm_result.get("input_tokens", 0),
            "output_tokens": llm_result.get("output_tokens", 0),
        }
    
    def batch_answer(
        self,
        questions: List[str],
        include_baseline: bool = True,
        top_k: int = None,
        temperature: float = 0.1,
    ) -> List[Dict]:
        """
        Answer a batch of questions with both RAG and baseline modes.
        
        Args:
            questions: List of questions to answer
            include_baseline: Whether to also generate baseline answers
            top_k: Number of chunks to retrieve per question
            temperature: LLM generation temperature
            
        Returns:
            List of result dictionaries, each containing RAG and optionally
            baseline answers
        """
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:60]}...")
            
            # RAG answer
            rag_result = self.answer(question, top_k=top_k, temperature=temperature)
            
            entry = {
                "question": question,
                "rag_answer": rag_result["answer"],
                "rag_time": rag_result["total_time"],
                "rag_chunks": rag_result["num_chunks_retrieved"],
                "rag_details": rag_result,
            }
            
            # Baseline answer
            if include_baseline:
                baseline_result = self.answer_baseline(question, temperature=temperature)
                entry["baseline_answer"] = baseline_result["answer"]
                entry["baseline_time"] = baseline_result["total_time"]
                entry["baseline_details"] = baseline_result
            
            results.append(entry)
        
        return results
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the current pipeline configuration."""
        return {
            "vector_store": self.vector_store.get_collection_info(),
            "embedding_model": self.embedding_model.get_info(),
            "llm": self.llm_client.get_info(),
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "queries_processed": len(self.query_history),
        }
    
    def save_results(self, results: List[Dict], output_path: Path):
        """
        Save batch results to a JSON file.
        
        Args:
            results: List of result dictionaries from batch_answer()
            output_path: Path to save the JSON file
        """
        clean_results = results
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "pipeline_info": self.get_pipeline_info(),
                "results": clean_results,
            }, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")

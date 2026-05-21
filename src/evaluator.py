"""
Evaluation Module for the RAG Q&A System.

Implements evaluation metrics for comparing RAG and baseline approaches.
Includes RAGAS-inspired metrics (faithfulness, answer relevancy, context
precision/recall) and custom metrics (ROUGE-L, semantic similarity).
"""

import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from src.config import METRICS_DIR, RANDOM_SEED, logger


def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score between prediction and reference texts.
    
    ROUGE-L measures the longest common subsequence (LCS) between
    the two texts, providing a measure of structural similarity.
    
    Args:
        prediction: Generated answer text
        reference: Ground truth answer text
        
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    # Tokenize into words
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Compute LCS length using dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    precision = lcs_length / m if m > 0 else 0.0
    recall = lcs_length / n if n > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_answer_similarity(
    prediction: str,
    reference: str,
    embedding_model=None,
) -> float:
    """
    Compute semantic similarity between prediction and reference using embeddings.
    
    Uses the embedding model to generate vectors for both texts and
    computes cosine similarity.
    
    Args:
        prediction: Generated answer text
        reference: Ground truth answer text
        embedding_model: Optional EmbeddingModel instance
        
    Returns:
        Cosine similarity score (0-1)
    """
    if embedding_model is None:
        # Fall back to word overlap (Jaccard similarity)
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        
        if not pred_words or not ref_words:
            return 0.0
        
        intersection = pred_words & ref_words
        union = pred_words | ref_words
        
        return round(len(intersection) / len(union), 4)
    
    # Use embedding-based similarity
    from src.embeddings import compute_similarity
    
    pred_embedding = embedding_model.embed_texts([prediction], show_progress=False)[0]
    ref_embedding = embedding_model.embed_texts([reference], show_progress=False)[0]
    
    return round(compute_similarity(pred_embedding, ref_embedding), 4)


def compute_faithfulness_simple(answer: str, context: str) -> float:
    """
    Compute a simple faithfulness score.
    
    Measures what fraction of content words in the answer appear in
    the provided context. Higher score means the answer is more
    grounded in the retrieved context.
    
    Args:
        answer: Generated answer text
        context: Retrieved context text
        
    Returns:
        Faithfulness score (0-1)
    """
    # Remove stop words for more meaningful comparison
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "and", "but", "or", "nor", "not", "so",
        "yet", "both", "either", "neither", "each", "every", "all",
        "any", "few", "more", "most", "other", "some", "such", "no",
        "only", "own", "same", "than", "too", "very", "it", "its",
        "this", "that", "these", "those", "i", "me", "my", "we",
        "our", "you", "your", "he", "him", "his", "she", "her",
        "they", "them", "their", "what", "which", "who", "whom",
    }
    
    answer_words = set(answer.lower().split()) - stop_words
    context_words = set(context.lower().split()) - stop_words
    
    if not answer_words:
        return 1.0  # Empty answer is trivially faithful
    
    grounded_words = answer_words & context_words
    
    return round(len(grounded_words) / len(answer_words), 4)


def compute_context_precision(
    retrieved_chunks: List[Dict],
    ground_truth_answer: str,
) -> float:
    """
    Compute context precision: fraction of retrieved chunks that are relevant.
    
    A chunk is considered relevant if it shares significant word overlap
    with the ground truth answer.
    
    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        ground_truth_answer: The correct answer text
        
    Returns:
        Precision score (0-1)
    """
    if not retrieved_chunks:
        return 0.0
    
    gt_words = set(ground_truth_answer.lower().split())
    relevant_count = 0
    
    for chunk in retrieved_chunks:
        chunk_text = chunk.get("text", "")
        chunk_words = set(chunk_text.lower().split())
        
        # Consider relevant if >10% word overlap with ground truth
        overlap = len(gt_words & chunk_words)
        if overlap > len(gt_words) * 0.1:
            relevant_count += 1
    
    return round(relevant_count / len(retrieved_chunks), 4)


def compute_context_recall(
    retrieved_chunks: List[Dict],
    ground_truth_answer: str,
) -> float:
    """
    Compute context recall: fraction of ground truth information found in context.
    
    Measures how much of the ground truth answer's key information is
    covered by the retrieved chunks.
    
    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        ground_truth_answer: The correct answer text
        
    Returns:
        Recall score (0-1)
    """
    if not ground_truth_answer.strip():
        return 1.0
    
    gt_words = set(ground_truth_answer.lower().split())
    
    # Combine all retrieved chunk texts
    all_context = " ".join(
        chunk.get("text", "") for chunk in retrieved_chunks
    )
    context_words = set(all_context.lower().split())
    
    if not gt_words:
        return 1.0
    
    covered = len(gt_words & context_words)
    
    return round(covered / len(gt_words), 4)


def evaluate_single(
    question: str,
    rag_answer: str,
    baseline_answer: str,
    ground_truth: str,
    context: str = "",
    retrieved_chunks: List[Dict] = None,
    embedding_model=None,
) -> Dict:
    """
    Evaluate a single question-answer pair for both RAG and baseline.
    
    Computes all metrics for a given question and returns a comprehensive
    evaluation dictionary.
    
    Args:
        question: The question asked
        rag_answer: Answer generated by RAG pipeline
        baseline_answer: Answer generated by baseline (no retrieval)
        ground_truth: Ground truth correct answer
        context: Formatted context string from retrieval
        retrieved_chunks: List of retrieved chunk dictionaries
        embedding_model: Optional embedding model for semantic similarity
        
    Returns:
        Dictionary with all evaluation metrics for both RAG and baseline
    """
    retrieved_chunks = retrieved_chunks or []
    
    # RAG metrics
    rag_rouge = compute_rouge_l(rag_answer, ground_truth)
    rag_similarity = compute_answer_similarity(rag_answer, ground_truth, embedding_model)
    rag_faithfulness = compute_faithfulness_simple(rag_answer, context) if context else 0.0
    ctx_precision = compute_context_precision(retrieved_chunks, ground_truth)
    ctx_recall = compute_context_recall(retrieved_chunks, ground_truth)
    
    # Baseline metrics
    baseline_rouge = compute_rouge_l(baseline_answer, ground_truth)
    baseline_similarity = compute_answer_similarity(baseline_answer, ground_truth, embedding_model)
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "rag": {
            "answer": rag_answer,
            "rouge_l_f1": rag_rouge["f1"],
            "rouge_l_precision": rag_rouge["precision"],
            "rouge_l_recall": rag_rouge["recall"],
            "answer_similarity": rag_similarity,
            "faithfulness": rag_faithfulness,
            "context_precision": ctx_precision,
            "context_recall": ctx_recall,
        },
        "baseline": {
            "answer": baseline_answer,
            "rouge_l_f1": baseline_rouge["f1"],
            "rouge_l_precision": baseline_rouge["precision"],
            "rouge_l_recall": baseline_rouge["recall"],
            "answer_similarity": baseline_similarity,
        },
    }


def evaluate_batch(
    results: List[Dict],
    ground_truths: Dict[str, str],
    embedding_model=None,
) -> Dict:
    """
    Evaluate a batch of RAG vs baseline results.
    
    Args:
        results: List of results from RAGPipeline.batch_answer()
        ground_truths: Dict mapping questions to ground truth answers
        embedding_model: Optional embedding model for semantic similarity
        
    Returns:
        Dictionary with per-question and aggregate evaluation metrics
    """
    evaluations = []
    
    for result in results:
        question = result["question"]
        gt = ground_truths.get(question, "")
        
        if not gt:
            logger.warning(f"No ground truth for: {question[:50]}...")
            continue
        
        # Get context and chunks from RAG details
        rag_details = result.get("rag_details", {})
        context = rag_details.get("context", "")
        chunks = rag_details.get("retrieved_chunks", [])
        
        evaluation = evaluate_single(
            question=question,
            rag_answer=result.get("rag_answer", ""),
            baseline_answer=result.get("baseline_answer", ""),
            ground_truth=gt,
            context=context,
            retrieved_chunks=chunks,
            embedding_model=embedding_model,
        )
        evaluations.append(evaluation)
    
    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(evaluations)
    
    return {
        "per_question": evaluations,
        "aggregate": aggregate,
        "num_evaluated": len(evaluations),
    }


def compute_aggregate_metrics(evaluations: List[Dict]) -> Dict:
    """
    Compute aggregate (mean) metrics across all evaluations.
    
    Args:
        evaluations: List of per-question evaluation dictionaries
        
    Returns:
        Dictionary with mean metrics for RAG and baseline
    """
    if not evaluations:
        return {}
    
    # Collect RAG metrics
    rag_metrics = defaultdict(list)
    baseline_metrics = defaultdict(list)
    
    for eval_item in evaluations:
        for key, value in eval_item["rag"].items():
            if isinstance(value, (int, float)):
                rag_metrics[key].append(value)
        
        for key, value in eval_item["baseline"].items():
            if isinstance(value, (int, float)):
                baseline_metrics[key].append(value)
    
    # Compute means
    aggregate = {
        "rag": {
            key: round(np.mean(values), 4)
            for key, values in rag_metrics.items()
        },
        "baseline": {
            key: round(np.mean(values), 4)
            for key, values in baseline_metrics.items()
        },
    }
    
    # Add improvement percentages
    improvements = {}
    for key in aggregate["rag"]:
        if key in aggregate["baseline"]:
            rag_val = aggregate["rag"][key]
            base_val = aggregate["baseline"][key]
            if base_val > 0:
                improvement = ((rag_val - base_val) / base_val) * 100
                improvements[key] = round(improvement, 2)
    
    aggregate["improvements_pct"] = improvements
    
    return aggregate


def save_evaluation_results(
    evaluation: Dict,
    experiment_name: str,
    output_dir: Path = None,
) -> Path:
    """
    Save evaluation results to a JSON file.
    
    Args:
        evaluation: Evaluation results dictionary
        experiment_name: Name for the experiment
        output_dir: Output directory (defaults to METRICS_DIR)
        
    Returns:
        Path to the saved file
    """
    output_dir = output_dir or METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"eval_{experiment_name}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {output_path}")
    return output_path


def load_ground_truth(gt_path: Path) -> Dict[str, str]:
    """
    Load ground truth Q&A pairs from a JSON file.
    
    Expected format:
    [
        {"question": "...", "answer": "...", "source": "..."},
        ...
    ]
    
    Args:
        gt_path: Path to the ground truth JSON file
        
    Returns:
        Dictionary mapping questions to ground truth answers
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    
    gt_dict = {}
    for pair in qa_pairs:
        gt_dict[pair["question"]] = pair["answer"]
    
    logger.info(f"Loaded {len(gt_dict)} ground truth Q&A pairs")
    return gt_dict


def print_evaluation_summary(aggregate: Dict):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        aggregate: Aggregate metrics dictionary from evaluate_batch()
    """
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY: RAG vs Baseline")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'RAG':>10} {'Baseline':>10} {'Improvement':>12}")
    print("-" * 60)
    
    rag = aggregate.get("rag", {})
    baseline = aggregate.get("baseline", {})
    improvements = aggregate.get("improvements_pct", {})
    
    # All metrics from RAG
    for metric in rag:
        rag_val = rag.get(metric, "N/A")
        base_val = baseline.get(metric, "N/A")
        imp_val = improvements.get(metric, "N/A")
        
        rag_str = f"{rag_val:.4f}" if isinstance(rag_val, float) else str(rag_val)
        base_str = f"{base_val:.4f}" if isinstance(base_val, float) else str(base_val)
        imp_str = f"{imp_val:+.1f}%" if isinstance(imp_val, float) else str(imp_val)
        
        print(f"  {metric:<23} {rag_str:>10} {base_str:>10} {imp_str:>12}")
    
    print("=" * 70)

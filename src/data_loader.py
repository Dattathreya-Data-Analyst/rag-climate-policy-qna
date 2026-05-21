"""
Data Loader Module for the RAG Q&A System.

Handles downloading, loading, and preprocessing of documents for the
Irish Climate Action Policy domain. Supports PDF and text file formats.
"""

import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, logger


# =============================================================================
# Dataset Sources - Irish Climate Action Policy Documents
# =============================================================================
# Publicly available documents from Irish government sources
DOCUMENT_SOURCES = {
    "climate_action_plan_2024": {
        "url": "https://www.gov.ie/pdf/?file=https://assets.gov.ie/285317/fbbaboratory-climate-action-plan-2024.pdf",
        "fallback_url": None,
        "filename": "climate_action_plan_2024.pdf",
        "title": "Ireland Climate Action Plan 2024",
        "source": "Government of Ireland (gov.ie)",
        "description": "Ireland's national climate action plan outlining government strategy for emissions reduction and climate targets.",
    },
    "epa_ghg_report": {
        "url": "https://www.epa.ie/publications/monitoring--assessment/climate-change/air-emissions/",
        "fallback_url": None,
        "filename": "epa_ghg_projections_2023.pdf",
        "title": "EPA Ireland GHG Emissions Projections",
        "source": "Environmental Protection Agency Ireland (epa.ie)",
        "description": "EPA greenhouse gas emissions data and projections for Ireland.",
    },
}


def download_file(url: str, save_path: Path, timeout: int = 60) -> bool:
    """
    Download a file from a URL and save it to the specified path.
    
    Args:
        url: URL to download from
        save_path: Local path to save the downloaded file
        timeout: Request timeout in seconds
        
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        logger.info(f"Downloading: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as f:
            if total_size > 0:
                with tqdm(total=total_size, unit="B", unit_scale=True, 
                         desc=save_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logger.info(f"Saved: {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        return False


def download_all_documents() -> Dict[str, bool]:
    """
    Download all configured document sources.
    
    Returns:
        Dictionary mapping document IDs to download success status
    """
    results = {}
    for doc_id, doc_info in DOCUMENT_SOURCES.items():
        save_path = RAW_DATA_DIR / doc_info["filename"]
        if save_path.exists():
            logger.info(f"Already downloaded: {doc_info['filename']}")
            results[doc_id] = True
            continue
        
        success = download_file(doc_info["url"], save_path)
        if not success and doc_info.get("fallback_url"):
            logger.info(f"Trying fallback URL for {doc_id}...")
            success = download_file(doc_info["fallback_url"], save_path)
        
        results[doc_id] = success
    
    return results


def load_pdf(file_path: Path) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
    """
    text = ""
    try:
        # Try PyPDF2 first
        from PyPDF2 import PdfReader
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        logger.info(f"Loaded PDF with PyPDF2: {file_path.name} ({len(reader.pages)} pages)")
    except ImportError:
        try:
            # Fallback to pypdf
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            logger.info(f"Loaded PDF with pypdf: {file_path.name} ({len(reader.pages)} pages)")
        except ImportError:
            logger.error("No PDF library available. Install PyPDF2 or pypdf.")
            raise
    
    return text


def load_text_file(file_path: Path) -> str:
    """
    Load a plain text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as string
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    logger.info(f"Loaded text file: {file_path.name} ({len(text)} chars)")
    return text


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Performs the following cleaning operations:
    - Remove excessive whitespace
    - Fix encoding artifacts
    - Remove page numbers and headers/footers
    - Normalize line breaks
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned and normalized text
    """
    # Replace common encoding artifacts
    text = text.replace("\x00", "")
    text = text.replace("\ufeff", "")
    
    # Remove standalone page numbers (e.g., "Page 1 of 4", or just "12")
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    
    # Normalize whitespace - collapse multiple spaces to single
    text = re.sub(r"[ \t]+", " ", text)
    
    # Normalize line breaks - collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    # Remove empty lines at start/end
    text = text.strip()
    
    return text


def load_document(file_path: Path) -> Dict:
    """
    Load and preprocess a single document.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary with keys: 'content', 'metadata', 'file_path'
    """
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf":
        raw_text = load_pdf(file_path)
    elif suffix in [".txt", ".md", ".text"]:
        raw_text = load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    # Clean the extracted text
    cleaned_text = clean_text(raw_text)
    
    # Build metadata
    metadata = {
        "source_file": file_path.name,
        "file_type": suffix,
        "raw_length": len(raw_text),
        "cleaned_length": len(cleaned_text),
        "word_count": len(cleaned_text.split()),
    }
    
    # Check if this document has source info in our config
    for doc_id, doc_info in DOCUMENT_SOURCES.items():
        if doc_info["filename"] == file_path.name:
            metadata.update({
                "title": doc_info["title"],
                "source": doc_info["source"],
                "description": doc_info["description"],
            })
            break
    
    return {
        "content": cleaned_text,
        "metadata": metadata,
        "file_path": str(file_path),
    }


def load_all_documents(data_dir: Path = None) -> List[Dict]:
    """
    Load all documents from the raw data directory.
    
    Args:
        data_dir: Directory to load documents from (defaults to RAW_DATA_DIR)
        
    Returns:
        List of document dictionaries
    """
    data_dir = data_dir or RAW_DATA_DIR
    documents = []
    
    # Supported file extensions
    extensions = {".pdf", ".txt", ".md", ".text"}
    
    for file_path in sorted(data_dir.iterdir()):
        if file_path.suffix.lower() in extensions:
            try:
                doc = load_document(file_path)
                documents.append(doc)
                logger.info(
                    f"  Loaded: {file_path.name} "
                    f"({doc['metadata']['word_count']} words)"
                )
            except Exception as e:
                logger.error(f"  Failed to load {file_path.name}: {e}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def save_processed_documents(documents: List[Dict], output_dir: Path = None):
    """
    Save processed documents as individual text files and a combined metadata JSON.
    
    Args:
        documents: List of document dictionaries from load_all_documents()
        output_dir: Directory to save processed files (defaults to PROCESSED_DATA_DIR)
    """
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metadata = []
    
    for i, doc in enumerate(documents):
        # Save cleaned text
        text_filename = f"doc_{i:02d}_{Path(doc['file_path']).stem}.txt"
        text_path = output_dir / text_filename
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(doc["content"])
        
        # Track metadata
        doc["metadata"]["processed_file"] = text_filename
        all_metadata.append(doc["metadata"])
    
    # Save combined metadata
    metadata_path = output_dir / "documents_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)
    
    logger.info(f"Saved {len(documents)} processed documents to {output_dir}")


def get_corpus_stats(documents: List[Dict]) -> Dict:
    """
    Compute statistics about the document corpus.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Dictionary with corpus statistics
    """
    total_chars = sum(len(doc["content"]) for doc in documents)
    total_words = sum(doc["metadata"]["word_count"] for doc in documents)
    
    stats = {
        "num_documents": len(documents),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_words_per_doc": total_words / len(documents) if documents else 0,
        "documents": [
            {
                "name": doc["metadata"].get("title", doc["metadata"]["source_file"]),
                "words": doc["metadata"]["word_count"],
                "chars": doc["metadata"]["cleaned_length"],
            }
            for doc in documents
        ],
    }
    
    return stats

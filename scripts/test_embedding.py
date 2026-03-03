"""Test script for Voyage-4 Embedding and FAISS VectorDB."""

import os
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import faiss
import numpy as np

# Load casadei core imports
from casadei.media import MediaBundle, TextMedia
from casadei.providers.voyage_embedding import VoyageEmbeddingProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
DB_INDEX_PATH = DATA_DIR / "vectordb.index"
DB_META_PATH = DATA_DIR / "vectordb_meta.json"
VOYAGE_DIMENSION = 1024  # Voyage-4 dimension


class VectorDB:
    """Lightweight FAISS Vector Database with persistence.
    
    Includes a query cache to optimize API costs by avoiding re-embedding 
    identical queries or documents.
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.texts: dict[int, str] = {}
        self.query_cache: dict[str, list[float]] = {}  # Cache text to its embedding
        self.next_id = 0
        
        # Load or create
        if DB_INDEX_PATH.exists() and DB_META_PATH.exists():
            logger.info("Loading existing FAISS index and metadata...")
            self.index = faiss.read_index(str(DB_INDEX_PATH))
            with open(DB_META_PATH, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                self.texts = {int(k): v for k, v in meta.get("texts", {}).items()}
                self.next_id = meta.get("next_id", 0)
                self.query_cache = meta.get("query_cache", {})
        else:
            logger.info("Creating new FAISS index...")
            # We use IndexFlatIP for cosine similarity (voyage produces normalized embeddings)
            self.index = faiss.IndexFlatIP(VOYAGE_DIMENSION)
            self.index = faiss.IndexIDMap(self.index)
            
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index successfully moved to GPU.")
            except AttributeError:
                logger.warning("faiss-gpu not found or no GPU available. Falling back to CPU.")
                self.use_gpu = False

    def add_documents(self, embeddings: list[list[float]], texts: list[str]) -> None:
        """Add embeddings and their text content to the database."""
        if not embeddings:
            return
            
        embs_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embs_np)
        
        ids = np.arange(self.next_id, self.next_id + len(texts), dtype=np.int64)
        
        self.index.add_with_ids(embs_np, ids)
        
        for i, text, emb in zip(ids, texts, embeddings):
            self.texts[int(i)] = text
            self.query_cache[text] = emb # Also cache the document text itself
            
        self.next_id += len(texts)
        logger.info(f"Added {len(texts)} documents to the vector DB.")
        self.save()

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        """Search the vectors for the query and return (text, score)."""
        if self.index.ntotal == 0:
            return []
            
        q_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q_np)
        
        distances, indices = self.index.search(q_np, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.texts:
                results.append((self.texts[int(idx)], float(dist)))
                
        return results

    def save(self) -> None:
        """Persist index and metadata."""
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        
        # Must move index back to CPU to serialize
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(DB_INDEX_PATH))
        else:
            faiss.write_index(self.index, str(DB_INDEX_PATH))
            
        with open(DB_META_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                "next_id": self.next_id,
                "texts": self.texts,
                "query_cache": self.query_cache
            }, f, indent=2)
            
        logger.info("Database saved to disk.")


def main():
    load_dotenv()
    
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        logger.error("VOYAGE_API_KEY missing from environment or .env file.")
        return

    # Check architecture notice
    logger.info("NOTE: linux-aarch64 detected; using faiss-cpu as GPU packages are unavailable.")
    db = VectorDB(use_gpu=False)
    
    provider = VoyageEmbeddingProvider(api_key=api_key)
    
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning models require lots of GPU memory.",
        "A recipe for chocolate chip cookies includes flour, sugar, and chocolate chips.",
        "France's capital city is Paris, known for the Eiffel Tower.",
        "I love building scalable software architectures."
    ]
    
    # Filter documents to only embed ones not in our proxy cache
    docs_to_embed = [d for d in documents if d not in db.query_cache]
    
    if docs_to_embed:
        logger.info(f"Embedding {len(docs_to_embed)} new documents...")
        items = {f"doc_{i}": TextMedia(text=doc) for i, doc in enumerate(docs_to_embed)}
        bundle = MediaBundle(items=items)
        
        out_bundle = provider.run(bundle)
        embeddings = out_bundle["embeddings"].embeddings
        db.add_documents(embeddings, docs_to_embed)
    else:
        logger.info("All documents are already cached in VectorDB. Skipping embedding.")
        
    # Test Search Query
    query = "Where is the Eiffel Tower?"
    logger.info(f"Querying: '{query}'")
    
    if query in db.query_cache:
        logger.info("Query found in cache, skipping API call.")
        query_emb = db.query_cache[query]
    else:
        logger.info("Query not in cache, invoking Voyage API...")
        query_bundle = MediaBundle(items={"query": TextMedia(text=query)})
        out_bundle = provider.run(query_bundle)
        query_emb = out_bundle["embeddings"].embeddings[0]
        # Cache the query embedding for next time
        db.query_cache[query] = query_emb
        db.save()
    
    results = db.search(query_emb, top_k=2)
    
    print("\n--- FAISS Search Results ---")
    print(f"Query: {query}\n")
    for text, score in results:
        print(f"Score: {score:.4f} | Document: {text}")
        
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()

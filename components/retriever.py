"""
retriever.py - RAG Retriever using FAISS + sentence-transformers
Embeds KB articles and performs semantic similarity search.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Optional

KB_PATH = os.path.join(os.path.dirname(__file__), "../data/knowledge_base.json")
EMBED_MODEL = "all-MiniLM-L6-v2"   # Fast, lightweight, good quality
TOP_K = 4

@dataclass
class RetrievedChunk:
    id: str
    title: str
    category: str
    content: str
    score: float
    tags: List[str]


class KBRetriever:
    """
    Loads the knowledge base, builds a FAISS index of embeddings,
    and retrieves top-K semantically similar chunks for a query.
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.articles = []
        self.index = None
        self.embeddings = None
        self._load_and_index()

    def _load_and_index(self):
        """Load KB JSON, embed all articles, build FAISS index."""
        with open(KB_PATH, "r") as f:
            self.articles = json.load(f)

        texts = [f"{a['title']}. {a['content']}" for a in self.articles]
        print(f"[Retriever] Embedding {len(texts)} KB articles...")
        self.embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)          # Inner-product = cosine (embeddings are normalized)
        self.index.add(self.embeddings.astype(np.float32))
        print(f"[Retriever] FAISS index built. Dim={dim}")

    def retrieve(self, query: str, category_filter: Optional[str] = None, top_k: int = TOP_K) -> List[RetrievedChunk]:
        """
        Embeds the query, searches FAISS index, optionally filters by category.
        Returns top-K RetrievedChunk objects.
        """
        query_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k * 3, len(self.articles)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            article = self.articles[idx]
            if category_filter and category_filter not in ("general",) and article["category"] != category_filter:
                continue
            results.append(RetrievedChunk(
                id=article["id"],
                title=article["title"],
                category=article["category"],
                content=article["content"],
                score=float(score),
                tags=article.get("tags", [])
            ))
            if len(results) >= top_k:
                break

        # If category filter returned too few results, fall back to no filter
        if len(results) < 2 and category_filter:
            return self.retrieve(query, category_filter=None, top_k=top_k)

        return results


# Singleton — loaded once when the app starts
_retriever_instance = None

def get_retriever() -> KBRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = KBRetriever()
    return _retriever_instance

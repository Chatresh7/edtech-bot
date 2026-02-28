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
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4


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

        # Build rich text for embedding: title + tags + content
        texts = []
        for a in self.articles:
            tags_str = ", ".join(a.get("tags", []))
            text = f"{a['title']}. Tags: {tags_str}. {a['content']}"
            texts.append(text)

        print(f"[Retriever] Embedding {len(texts)} KB articles...")
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32
        )

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner-product = cosine (normalized vectors)
        self.index.add(self.embeddings.astype(np.float32))
        print(f"[Retriever] FAISS index built. {len(texts)} articles indexed. Dim={dim}")

    def retrieve(
        self,
        query: str,
        category_filter: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K
    ) -> List[RetrievedChunk]:
        """
        Embeds the query, searches FAISS index.
        Returns top-K RetrievedChunk objects sorted by relevance score.

        Strategy:
        1. Fetch top_k * 4 candidates from FAISS
        2. If category_filter is valid and specific, prefer those articles
        3. Always fill remaining slots with next best results (no hard cut)
        4. Return exactly top_k results
        """

        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)

        # Fetch a large candidate pool
        pool_size = min(top_k * 6, len(self.articles))
        scores, indices = self.index.search(query_vec, pool_size)

        all_candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            article = self.articles[idx]
            all_candidates.append(RetrievedChunk(
                id=article["id"],
                title=article["title"],
                category=article["category"],
                content=article["content"],
                score=float(score),
                tags=article.get("tags", [])
            ))

        # Apply soft category preference (not hard filter)
        valid_categories = {"course", "assessment", "certification", "progress"}

        if category_filter and category_filter in valid_categories:
            # Split into preferred (matching category) and others
            preferred = [c for c in all_candidates if c.category == category_filter]
            others = [c for c in all_candidates if c.category != category_filter]

            # Take as many preferred as available, fill rest with others
            # But only prefer if score is reasonably high (> 0.2)
            preferred_high = [c for c in preferred if c.score > 0.20]
            others_high = [c for c in others if c.score > 0.20]

            # Combine: prefer category matches but don't exclude good cross-category results
            if len(preferred_high) >= top_k:
                # Enough high-quality matches in preferred category
                results = preferred_high[:top_k]
            elif len(preferred_high) > 0:
                # Some preferred, fill rest from others
                needed = top_k - len(preferred_high)
                results = preferred_high + others_high[:needed]
            else:
                # No good preferred matches — use pure similarity ranking
                results = all_candidates[:top_k]
        else:
            # No category filter — pure similarity ranking
            results = all_candidates[:top_k]

        # Final sort by score descending
        results = sorted(results, key=lambda x: x.score, reverse=True)

        # Ensure we return exactly top_k (or fewer if not enough results)
        return results[:top_k]

    def get_stats(self):
        """Return KB statistics."""
        from collections import Counter
        cats = Counter(a["category"] for a in self.articles)
        return {
            "total_articles": len(self.articles),
            "by_category": dict(cats)
        }


# Singleton — loaded once when the app starts
_retriever_instance = None


def get_retriever() -> KBRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = KBRetriever()
    return _retriever_instance
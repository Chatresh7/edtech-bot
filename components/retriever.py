"""
retriever.py - RAG Retriever using FAISS + sentence-transformers
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Optional
from collections import Counter

KB_PATH    = os.path.join(os.path.dirname(__file__), "../data/knowledge_base.json")
EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    id:       str
    title:    str
    category: str
    content:  str
    score:    float
    tags:     List[str]


class KBRetriever:
    def __init__(self):
        self.model    = SentenceTransformer(EMBED_MODEL)
        self.articles = []
        self.index    = None
        self._load_and_index()

    def _load_and_index(self):
        with open(KB_PATH, "r") as f:
            self.articles = json.load(f)

        # Richer embeddings: title + tags + content
        texts = []
        for a in self.articles:
            tags_str = ", ".join(a.get("tags", []))
            texts.append(f"{a['title']}. Tags: {tags_str}. {a['content']}")

        print(f"[Retriever] Embedding {len(texts)} articles...")
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )
        self.embeddings = embeddings.astype(np.float32)

        dim        = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # cosine on normalized vectors
        self.index.add(self.embeddings)
        print(f"[Retriever] Index ready. {len(texts)} articles, dim={dim}")

    def retrieve(
        self,
        query:           str,
        category_filter: Optional[str] = None,
        top_k:           int           = 4
    ) -> List[RetrievedChunk]:
        """
        Returns EXACTLY top_k chunks ranked by cosine similarity.
        Soft category preference: prefers matching category but fills
        remaining slots with best cross-category results.
        """
        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Fetch a large candidate pool (6x top_k, min 20)
        pool = min(max(top_k * 6, 20), len(self.articles))
        scores, indices = self.index.search(query_vec, pool)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            a = self.articles[idx]
            candidates.append(RetrievedChunk(
                id=a["id"], title=a["title"], category=a["category"],
                content=a["content"], score=float(score), tags=a.get("tags", [])
            ))

        valid_cats = {"course", "assessment", "certification", "progress"}

        if category_filter and category_filter in valid_cats:
            preferred = [c for c in candidates if c.category == category_filter]
            others    = [c for c in candidates if c.category != category_filter]

            if len(preferred) >= top_k:
                result = preferred[:top_k]
            else:
                result = preferred + others[: top_k - len(preferred)]
        else:
            result = candidates[:top_k]

        # Final sort by score
        result = sorted(result, key=lambda x: x.score, reverse=True)
        return result[:top_k]   # ALWAYS exactly top_k

    def get_stats(self) -> dict:
        cats = Counter(a["category"] for a in self.articles)
        return {"total_articles": len(self.articles), "by_category": dict(cats)}


_retriever_instance = None

def get_retriever() -> KBRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = KBRetriever()
    return _retriever_instance
from __future__ import annotations
import os
import sys
from dataclasses import replace
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TOP_K
from parte1.dense_retriever import DenseRetriever, RetrievedChunk
from utils.mdb_client import MDBClient, get_client

from part3.sparse_index import tokenize
from part3.sparse_retriever import SparseRetriever


def _chunk_key(chunk: RetrievedChunk) -> str:
    return chunk.chunk_id or chunk.intervention_id


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class HybridRetriever:
    """
    Recuperador hibrido dense+sparse.

    Primero une rankings con Reciprocal Rank Fusion. Luego aplica un rerank
    deterministico que premia cobertura lexical de la pregunta y penaliza
    duplicados muy similares mediante diversidad tipo MMR.
    """

    def __init__(
        self,
        client: Optional[MDBClient] = None,
        dense: Optional[DenseRetriever] = None,
        sparse: Optional[SparseRetriever] = None,
        top_k: int = TOP_K,
        pool_multiplier: int = 4,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        diversity_lambda: float = 0.82,
    ) -> None:
        self.client = client or get_client()
        self.top_k = top_k
        self.pool_multiplier = pool_multiplier
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.diversity_lambda = diversity_lambda
        self.dense = dense or DenseRetriever(client=self.client, top_k=top_k)
        self.sparse = sparse or SparseRetriever(client=self.client, top_k=top_k)

    def retrieve(self, question: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        pool_k = max(k * self.pool_multiplier, k, 20)

        dense_chunks = self.dense.retrieve(question, k=pool_k)
        sparse_chunks = self.sparse.retrieve(question, k=pool_k)

        candidates: dict[str, RetrievedChunk] = {}
        features: dict[str, dict] = {}

        for rank, chunk in enumerate(dense_chunks, start=1):
            key = _chunk_key(chunk)
            if not key:
                continue
            candidates.setdefault(key, chunk)
            data = features.setdefault(key, {})
            data["dense_rank"] = rank
            data["dense_distance"] = chunk.score
            data["rrf_score"] = data.get("rrf_score", 0.0) + self.dense_weight / (self.rrf_k + rank)

        for rank, chunk in enumerate(sparse_chunks, start=1):
            key = _chunk_key(chunk)
            if not key:
                continue
            candidates.setdefault(key, chunk)
            data = features.setdefault(key, {})
            data["sparse_rank"] = rank
            data["sparse_score"] = chunk.score
            data["matched_terms"] = chunk.extra.get("matched_terms", [])
            data["rrf_score"] = data.get("rrf_score", 0.0) + self.sparse_weight / (self.rrf_k + rank)

        if not candidates:
            return []

        query_terms = set(tokenize(question))
        token_cache = {key: set(tokenize(chunk.text)) for key, chunk in candidates.items()}
        max_rrf = max(data.get("rrf_score", 0.0) for data in features.values()) or 1.0

        for key, data in features.items():
            matched = query_terms & token_cache[key]
            lexical_overlap = len(matched) / len(query_terms) if query_terms else 0.0
            base_score = data.get("rrf_score", 0.0) / max_rrf
            data["lexical_overlap"] = lexical_overlap
            data["matched_terms"] = sorted(set(data.get("matched_terms", [])) | matched)
            data["rerank_base"] = 0.78 * base_score + 0.22 * lexical_overlap

        selected: list[str] = []
        remaining = set(candidates)
        while remaining and len(selected) < k:
            best_key = None
            best_score = float("-inf")
            for key in remaining:
                novelty_penalty = 0.0
                if selected:
                    novelty_penalty = max(_jaccard(token_cache[key], token_cache[sel]) for sel in selected)
                score = (
                    self.diversity_lambda * features[key]["rerank_base"]
                    - (1.0 - self.diversity_lambda) * novelty_penalty
                )
                if score > best_score:
                    best_key = key
                    best_score = score

            if best_key is None:
                break
            features[best_key]["final_score"] = best_score
            selected.append(best_key)
            remaining.remove(best_key)

        output: list[RetrievedChunk] = []
        for final_rank, key in enumerate(selected, start=1):
            chunk = candidates[key]
            data = features[key]
            output.append(
                replace(
                    chunk,
                    score=data.get("final_score", data.get("rerank_base", 0.0)),
                    extra={
                        **chunk.extra,
                        "retriever": "hybrid_rrf_mmr",
                        "rank": final_rank,
                        "dense_rank": data.get("dense_rank"),
                        "dense_distance": data.get("dense_distance"),
                        "sparse_rank": data.get("sparse_rank"),
                        "sparse_score": data.get("sparse_score"),
                        "rrf_score": data.get("rrf_score", 0.0),
                        "lexical_overlap": data.get("lexical_overlap", 0.0),
                        "matched_terms": data.get("matched_terms", []),
                        "score_direction": "higher_is_better",
                    },
                )
            )

        return output

    def retrieve_as_context(self, question: str, k: Optional[int] = None) -> str:
        chunks = self.retrieve(question, k)
        if not chunks:
            return "No se encontro contexto relevante por recuperacion hibrida."
        return "\n\n---\n\n".join(chunk.to_context_str() for chunk in chunks)

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TOP_K
from parte1.dense_retriever import RetrievedChunk
from utils.mdb_client import MDBClient, get_client

from part3.sparse_index import BM25Index, DEFAULT_INDEX_PATH, SparseDocument


def _document_to_chunk(doc: SparseDocument, score: float, extra: Optional[dict] = None) -> RetrievedChunk:
    return RetrievedChunk(
        intervention_id=doc.intervention_id,
        chunk_id=doc.chunk_id,
        text=doc.text,
        score=score,
        speaker=doc.speaker,
        party=doc.party,
        chamber=doc.chamber,
        session_date=doc.session_date,
        extra=extra or {},
    )


class SparseRetriever:
    """Retriever BM25 compatible con el RAGPipeline existente."""

    def __init__(
        self,
        client: Optional[MDBClient] = None,
        top_k: int = TOP_K,
        index_path: Path = DEFAULT_INDEX_PATH,
        auto_build: bool = True,
        build_limit: Optional[int] = None,
    ) -> None:
        self.client = client or get_client()
        self.top_k = top_k
        self.index_path = Path(index_path)
        self.index = self._load_or_build(auto_build=auto_build, build_limit=build_limit)

    def _load_or_build(self, auto_build: bool, build_limit: Optional[int]) -> BM25Index:
        if self.index_path.exists():
            index = BM25Index.load(self.index_path)
            if index.size > 0 and index.vocabulary_size > 0:
                return index
            if not auto_build:
                raise ValueError(
                    f"El indice sparse en {self.index_path} esta vacio. "
                    "Reconstruyelo con: python part3/run_parte3.py --force-rebuild"
                )
            print(f"[Parte 3] Indice sparse vacio en {self.index_path}; reconstruyendo...")
            index = BM25Index.from_mdb(self.client, limit=build_limit)
            index.save(self.index_path)
            return index
        if not auto_build:
            raise FileNotFoundError(
                f"No existe el indice sparse en {self.index_path}. "
                "Ejecuta: python part3/run_parte3.py --build-index"
            )
        index = BM25Index.from_mdb(self.client, limit=build_limit)
        index.save(self.index_path)
        return index

    def rebuild(self, limit: Optional[int] = None) -> BM25Index:
        self.index = BM25Index.from_mdb(self.client, limit=limit)
        self.index.save(self.index_path)
        return self.index

    def retrieve(self, question: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        results = self.index.search(question, top_k=k)
        return [
            _document_to_chunk(
                result.document,
                result.score,
                extra={
                    "retriever": "sparse_bm25",
                    "rank": result.rank,
                    "matched_terms": result.matched_terms,
                    "score_direction": "higher_is_better",
                },
            )
            for result in results
        ]

    def retrieve_as_context(self, question: str, k: Optional[int] = None) -> str:
        chunks = self.retrieve(question, k)
        if not chunks:
            return "No se encontro contexto relevante por BM25."
        return "\n\n---\n\n".join(chunk.to_context_str() for chunk in chunks)

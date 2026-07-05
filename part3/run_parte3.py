from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TOP_K
from parte1.dense_retriever import DenseRetriever
from parte1.rag_pipeline import RAGPipeline
from part3.hybrid_retriever import HybridRetriever
from part3.sparse_index import DEFAULT_INDEX_PATH, BM25Index
from part3.sparse_retriever import SparseRetriever
from utils.mdb_client import MDBClient


QUESTIONS = [
    "Identifica intervenciones que mencionen explicitamente CAE, condonacion y FES en educacion superior.",
    "Como se discutio la Ley Nain-Retamal y la legitima defensa de Carabineros?",
    "Que se dijo sobre boletines o proyectos asociados a muerte digna, eutanasia y cuidados paliativos?",
]


def chunk_to_dict(chunk: Any) -> dict[str, Any]:
    return {
        "intervention_id": chunk.intervention_id,
        "chunk_id": chunk.chunk_id,
        "speaker": chunk.speaker,
        "party": chunk.party,
        "chamber": chunk.chamber,
        "session_date": chunk.session_date,
        "score": chunk.score,
        "extra": chunk.extra,
        "text": chunk.text,
    }


def answer_or_retrieve(pipeline: RAGPipeline, question: str, k: int, no_llm: bool) -> dict[str, Any]:
    if no_llm:
        chunks = pipeline.retriever.retrieve(question, k=k)
        return {
            "question": question,
            "answer": None,
            "retrieved_chunks": [chunk_to_dict(chunk) for chunk in chunks],
        }

    response = pipeline.answer(question, k=k)
    return {
        "question": response.question,
        "answer": response.answer,
        "retrieved_chunks": [chunk_to_dict(chunk) for chunk in response.retrieved_chunks],
    }


def build_index(client: MDBClient, limit: int | None, index_path: Path) -> BM25Index:
    print("[Parte 3] Construyendo indice BM25 desde MillenniumDB...")
    index = BM25Index.from_mdb(client, limit=limit)
    index.save(index_path)
    print(
        f"[Parte 3] Indice guardado en {index_path} "
        f"({index.size} documentos, {index.vocabulary_size} terminos)."
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 3: sparse BM25 e hibrido dense+sparse.")
    parser.add_argument("--build-index", action="store_true", help="Reconstruye el store BM25 y termina si se usa con --build-only.")
    parser.add_argument("--build-only", action="store_true", help="Solo construye el indice sparse.")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignora el indice existente y lo reconstruye.")
    parser.add_argument("--limit", type=int, default=None, help="Limite opcional de documentos para pruebas rapidas.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-k final para cada retriever.")
    parser.add_argument("--no-llm", action="store_true", help="No llama a OpenAI; solo guarda rankings recuperados.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH, help="Ruta del indice sparse serializado.")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent / f"resultados_parte3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print("=" * 70)
    print("PARTE 3 - Recuperacion hibrida (sparse + denso)")
    print(f"Top-K: {args.k}")
    print(f"Indice: {args.index_path}")
    print(f"LLM: {'desactivado' if args.no_llm else 'gpt-4o-mini'}")
    print("=" * 70)

    with MDBClient() as client:
        should_build = args.build_index or args.force_rebuild or not args.index_path.exists()
        if should_build:
            build_index(client, limit=args.limit, index_path=args.index_path)

        if args.build_only:
            return

        sparse = SparseRetriever(
            client=client,
            top_k=args.k,
            index_path=args.index_path,
            auto_build=True,
            build_limit=args.limit,
        )
        dense = DenseRetriever(client=client, top_k=args.k)
        hybrid = HybridRetriever(client=client, dense=dense, sparse=sparse, top_k=args.k)

        sparse_pipeline = RAGPipeline(sparse, retriever_name="sparse_bm25", top_k=args.k)
        hybrid_pipeline = RAGPipeline(hybrid, retriever_name="hybrid_rrf_mmr", top_k=args.k)

        log: dict[str, Any] = {
            "parte": 3,
            "timestamp": datetime.now().isoformat(),
            "top_k": args.k,
            "sparse_strategy": "BM25 propio",
            "fusion": "Reciprocal Rank Fusion",
            "rerank": "MMR/Jaccard deterministico, sin LLM",
            "index_path": str(args.index_path),
            "questions": [],
        }

        for idx, question in enumerate(QUESTIONS, start=1):
            print(f"\n[{idx}/{len(QUESTIONS)}] {question}")
            sparse_result = answer_or_retrieve(sparse_pipeline, question, args.k, args.no_llm)
            hybrid_result = answer_or_retrieve(hybrid_pipeline, question, args.k, args.no_llm)

            print("  Sparse:")
            for rank, chunk in enumerate(sparse_result["retrieved_chunks"], start=1):
                print(f"    {rank}. INT-{chunk['intervention_id']} score={chunk['score']:.4f}")

            print("  Hibrido:")
            for rank, chunk in enumerate(hybrid_result["retrieved_chunks"], start=1):
                dense_rank = chunk["extra"].get("dense_rank")
                sparse_rank = chunk["extra"].get("sparse_rank")
                print(
                    f"    {rank}. INT-{chunk['intervention_id']} "
                    f"score={chunk['score']:.4f} dense_rank={dense_rank} sparse_rank={sparse_rank}"
                )

            log["questions"].append(
                {
                    "question": question,
                    "sparse": sparse_result,
                    "hybrid": hybrid_result,
                }
            )

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=2)

    print(f"\n[Parte 3] Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()

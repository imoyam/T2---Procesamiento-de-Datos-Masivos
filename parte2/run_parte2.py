from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    SYSTEM_PROMPT,
    TOP_K,
    USER_PROMPT_TEMPLATE,
)
from parte1.dense_retriever import DenseRetriever, RetrievedChunk
from parte2.graph_retriever import GraphRetriever
from utils.mdb_client import MDBClient

K = TOP_K
LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"resultados_parte2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
)


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def chunk_to_dict(chunk: RetrievedChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "intervention_id": chunk.intervention_id,
        "score": chunk.score,
        "speaker": chunk.speaker,
        "party": chunk.party,
        "chamber": chunk.chamber,
        "session_date": chunk.session_date,
        "birth_date": chunk.extra.get("birth_date", ""),
        "legislative_period": chunk.extra.get("legislative_period", ""),
        "text_preview": chunk.text[:300].replace("\n", " "),
    }


def chunks_to_log(chunks: Iterable[RetrievedChunk]) -> list[dict]:
    return [chunk_to_dict(chunk) for chunk in chunks]


def chunks_as_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No se encontraron intervenciones relevantes."
    return "\n\n---\n\n".join(chunk.to_context_str() for chunk in chunks)


def grouped_to_log(grouped: dict[str, list[RetrievedChunk]]) -> dict[str, list[dict]]:
    return {group: chunks_to_log(chunks) for group, chunks in grouped.items()}


def print_chunks(chunks: list[RetrievedChunk], label: str) -> None:
    print(f"\n[{label}] {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks, 1):
        print(
            f"  {i}. [INT-{chunk.intervention_id}] "
            f"chunk={chunk.chunk_id} score={chunk.score:.4f} "
            f"| {chunk.speaker} | {chunk.party} | {chunk.chamber}"
        )
        if chunk.session_date:
            print(f"     fecha: {chunk.session_date}")
        print(f"     {chunk.text[:130].replace(chr(10), ' ')}...")


def print_grouped(grouped: dict[str, list[RetrievedChunk]], label: str) -> None:
    print(f"\n[{label}]")
    for group, chunks in grouped.items():
        print(f"  -- {group} ({len(chunks)} chunk(s))")
        for chunk in chunks:
            print(f"     [INT-{chunk.intervention_id}] score={chunk.score:.4f} | {chunk.speaker}")


def answer_from_chunks(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No se genero respuesta: el patron no recupero intervenciones."
    if not OPENAI_API_KEY:
        return "No se genero respuesta: falta OPENAI_API_KEY."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        user_msg = USER_PROMPT_TEMPLATE.format(context=chunks_as_context(chunks), question=question)
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        return f"Error al generar respuesta con {LLM_MODEL}: {exc}"


def answer_from_grouped(question: str, grouped: dict[str, list[RetrievedChunk]]) -> str:
    chunks = [chunk for group_chunks in grouped.values() for chunk in group_chunks]
    return answer_from_chunks(question, chunks)


def demo_patron_a(graph: GraphRetriever) -> dict:
    print_section("Patron A - Recuperacion con restriccion tipada")
    resultados: dict[str, dict] = {}

    question_party = "Que opina el Partido Socialista de Chile sobre la reforma de pensiones?"
    party = "Partido Socialista de Chile"
    chunks_party = graph.retrieve_by_party(question_party, party_name=party, k=K)
    print_chunks(chunks_party, "Filtro por partido")
    resultados["partido"] = {
        "pattern": "semantic_search_filtered_by_party",
        "question": question_party,
        "party_filter": party,
        "answer": answer_from_chunks(question_party, chunks_party),
        "retrieved": chunks_to_log(chunks_party),
    }

    question_chamber = "Que se ha debatido sobre el acceso al agua?"
    chamber = "Senado"
    chunks_chamber = graph.retrieve_by_chamber(question_chamber, chamber_name=chamber, k=K)
    print_chunks(chunks_chamber, "Filtro por camara/rol")
    resultados["camara"] = {
        "pattern": "semantic_search_filtered_by_chamber_role",
        "question": question_chamber,
        "chamber_filter": chamber,
        "role_filter_used": "Senador",
        "answer": answer_from_chunks(question_chamber, chunks_chamber),
        "retrieved": chunks_to_log(chunks_chamber),
    }

    return resultados


def demo_patron_b(graph: GraphRetriever) -> dict:
    print_section("Patron B - Agregacion y contraste")
    resultados: dict[str, dict] = {}

    question_party = "Que posturas defienden los distintos partidos sobre migracion irregular y seguridad fronteriza?"
    grouped_party = graph.retrieve_contrast_party(question_party, k_per_party=2, top_parties=5)
    print_grouped(grouped_party, "Contraste por partido")
    resultados["contraste_partido"] = {
        "pattern": "semantic_search_grouped_by_party",
        "question": question_party,
        "groups_found": list(grouped_party.keys()),
        "answer": answer_from_grouped(question_party, grouped_party),
        "retrieved_by_group": grouped_to_log(grouped_party),
    }

    question_chamber = "Como se ha debatido el proyecto de ley de salud mental?"
    grouped_chamber = graph.retrieve_contrast_chamber(question_chamber, k_per_chamber=2)
    print_grouped(grouped_chamber, "Contraste por camara/rol")
    resultados["contraste_camara"] = {
        "pattern": "semantic_search_grouped_by_chamber_role",
        "question": question_chamber,
        "groups_found": list(grouped_chamber.keys()),
        "answer": answer_from_grouped(question_chamber, grouped_chamber),
        "retrieved_by_group": grouped_to_log(grouped_chamber),
    }

    return resultados


def demo_patron_c(graph: GraphRetriever) -> dict:
    print_section("Patron C - Atributo temporal o numerico")
    resultados: dict[str, dict] = {}

    question_age = "Que se dice sobre cambio climatico y medio ambiente?"
    young = graph.retrieve_by_age_cohort(question_age, birth_year_min=1975, birth_year_max=1995, k=K)
    older = graph.retrieve_by_age_cohort(question_age, birth_year_min=1940, birth_year_max=1960, k=K)
    print_chunks(young, "Cohorte nacida 1975-1995")
    print_chunks(older, "Cohorte nacida 1940-1960")
    resultados["cohorte_etaria"] = {
        "pattern": "semantic_search_filtered_by_birth_date",
        "question": question_age,
        "cohorts": {
            "1975_1995": {
                "answer": answer_from_chunks(
                    "Resume que dicen parlamentarios nacidos entre 1975 y 1995 sobre medio ambiente.",
                    young,
                ),
                "retrieved": chunks_to_log(young),
            },
            "1940_1960": {
                "answer": answer_from_chunks(
                    "Resume que dicen parlamentarios nacidos entre 1940 y 1960 sobre medio ambiente.",
                    older,
                ),
                "retrieved": chunks_to_log(older),
            },
        },
    }

    question_dates = "Que se ha discutido sobre violencia en La Araucania?"
    chunks_2025 = graph.retrieve_by_date_range(
        question_dates,
        date_start="2025-01-01T00:00:00",
        date_end="2025-12-31T23:59:59",
        k=K,
    )
    chunks_2026 = graph.retrieve_by_date_range(
        question_dates,
        date_start="2026-01-01T00:00:00",
        date_end="2026-12-31T23:59:59",
        k=K,
    )
    print_chunks(chunks_2025, "Rango temporal 2025")
    print_chunks(chunks_2026, "Rango temporal 2026")
    resultados["rango_fechas"] = {
        "pattern": "semantic_search_filtered_by_session_date",
        "question": question_dates,
        "ranges": {
            "2025": {
                "answer": answer_from_chunks(
                    "Resume las intervenciones de 2025 sobre violencia en La Araucania.",
                    chunks_2025,
                ),
                "retrieved": chunks_to_log(chunks_2025),
            },
            "2026": {
                "answer": answer_from_chunks(
                    "Resume las intervenciones de 2026 sobre violencia en La Araucania.",
                    chunks_2026,
                ),
                "retrieved": chunks_to_log(chunks_2026),
            },
        },
    }

    evolution = graph.retrieve_temporal_evolution(question_dates, k_per_year=2)
    print_grouped(evolution, "Evolucion temporal por anio")
    resultados["evolucion_temporal"] = {
        "pattern": "semantic_search_grouped_by_session_year",
        "question": question_dates,
        "retrieved_by_year": grouped_to_log(evolution),
    }

    return resultados


def demo_comparacion(dense: DenseRetriever, graph: GraphRetriever) -> dict:
    print_section("Parte 2.2 - Comparacion Dense RAG vs GraphRAG")
    question = "Que opina la UDI sobre el impuesto a las grandes fortunas?"
    party = "Partido Unión Demócrata Independiente"

    dense_chunks = dense.retrieve(question, k=K)
    graph_chunks = graph.retrieve_by_party(question, party_name=party, k=K)

    print_chunks(dense_chunks, "Denso sin filtro")
    print_chunks(graph_chunks, "GraphRAG con filtro UDI")

    dense_parties = sorted({chunk.party for chunk in dense_chunks if chunk.party})
    graph_parties = sorted({chunk.party for chunk in graph_chunks if chunk.party})

    return {
        "question": question,
        "party_filter": party,
        "why_graph_should_help": (
            "La pregunta pide una postura de partido. El denso optimiza similitud semantica "
            "y puede mezclar partidos; GraphRAG agrega el patron tipado Position -> PoliticalParty "
            "antes de ordenar por distancia coseno."
        ),
        "dense": {
            "parties_in_context": dense_parties,
            "answer": answer_from_chunks(question, dense_chunks),
            "retrieved": chunks_to_log(dense_chunks),
        },
        "graph": {
            "parties_in_context": graph_parties,
            "answer": answer_from_chunks(question, graph_chunks),
            "retrieved": chunks_to_log(graph_chunks),
        },
    }


def main() -> None:
    print("=" * 72)
    print("PARTE 2 - GraphRAG sobre MillenniumDB")
    print(f"Modelo LLM  : {LLM_MODEL}")
    print(f"Top-K       : {K}")
    print("Retriever   : grafo + distancia coseno sobre embeddings en MDB")
    print("=" * 72)

    log = {
        "parte": 2,
        "estrategia": "GraphRAG",
        "top_k": K,
        "llm_model": LLM_MODEL,
        "timestamp": datetime.now().isoformat(),
        "requirements_checked": [
            "restriccion tipada por partido",
            "restriccion tipada por camara/rol",
            "agregacion y contraste por partido",
            "agregacion y contraste por camara/rol",
            "atributo numerico/temporal por fecha de nacimiento",
            "atributo temporal por fecha de sesion",
            "comparacion GraphRAG vs RAG denso",
        ],
        "resultados": {},
    }

    with MDBClient() as client:
        dense = DenseRetriever(client=client, top_k=K, use_metadata=True)
        graph = GraphRetriever(client=client, top_k=K)

        for key, fn in (
            ("patron_a", lambda: demo_patron_a(graph)),
            ("patron_b", lambda: demo_patron_b(graph)),
            ("patron_c", lambda: demo_patron_c(graph)),
            ("comparacion_2_2", lambda: demo_comparacion(dense, graph)),
        ):
            try:
                log["resultados"][key] = fn()
            except Exception as exc:
                print(f"[ERROR] {key}: {exc}")
                log["resultados"][key] = {"error": str(exc)}

    with open(LOG_FILE, "w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Log guardado en: {LOG_FILE}")
    print("Parte 2 completada.")


if __name__ == "__main__":
    main()

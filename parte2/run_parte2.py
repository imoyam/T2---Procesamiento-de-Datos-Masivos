
from __future__ import annotations
import sys, os, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdb_client import MDBClient
from parte1.dense_retriever import DenseRetriever
from parte1.rag_pipeline import RAGPipeline
from parte2.graph_retriever import GraphRetriever
from config.settings import TOP_K

# ── Configuración ─────────────────────────────────────────────────────────────
K = TOP_K
LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"resultados_parte2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de presentación
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def print_chunks(chunks, label: str = "") -> None:
    if label:
        print(f"\n  [{label}] {len(chunks)} chunk(s) recuperado(s):")
    for i, c in enumerate(chunks, 1):
        print(f"    {i}. [INT-{c.intervention_id}] score={c.score:.4f}")
        print(f"       Orador : {c.speaker}")
        print(f"       Partido: {c.party}")
        if c.session_date:
            print(f"       Fecha  : {c.session_date}")
        preview = c.text[:120].replace("\n", " ")
        print(f"       Texto  : {preview}…")


def print_contrast(grouped: dict, label: str = "") -> None:
    if label:
        print(f"\n  [{label}]")
    for group_key, chunks in grouped.items():
        print(f"\n  ── {group_key} ──")
        for c in chunks:
            print(f"    [INT-{c.intervention_id}] score={c.score:.4f} | {c.speaker}")
            preview = c.text[:100].replace("\n", " ")
            print(f"    {preview}…")


# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN A — Restricción tipada
# ─────────────────────────────────────────────────────────────────────────────

def demo_patron_a(graph: GraphRetriever, pipeline_graph: RAGPipeline) -> dict:
    print_section("PATRÓN A — Recuperación con restricción tipada")
    resultados = {}

    # A.1 — Filtro por partido
    question_partido = "¿Cuál es la postura sobre la reforma de pensiones?"
    partido = "Partido Socialista de Chile"   # ← ajustar según los partidos en tus datos
    print(f"\n  A.1 Filtro por PARTIDO: '{partido}'")
    print(f"  Pregunta: {question_partido}")

    chunks_partido = graph.retrieve_by_party(question_partido, party_name=partido, k=K)
    print_chunks(chunks_partido, label="GraphRAG filtrado por partido")

    context_partido = "\n\n---\n\n".join(c.to_context_str() for c in chunks_partido)
    resp_partido = pipeline_graph.answer(
        f"¿Qué opina el {partido} sobre la reforma de pensiones?",
    )
    print(f"\n  RESPUESTA LLM:\n  {resp_partido.answer[:500]}…")
    resultados["patron_a_partido"] = {
        "question": resp_partido.question,
        "party_filter": partido,
        "answer": resp_partido.answer,
        "retrieved_ids": [c.intervention_id for c in chunks_partido],
    }

    # A.2 — Filtro por cámara
    question_camara = "¿Qué se ha debatido sobre el acceso al agua?"
    camara = "Senado"
    print(f"\n  A.2 Filtro por CÁMARA: '{camara}'")
    print(f"  Pregunta: {question_camara}")

    chunks_camara = graph.retrieve_by_chamber(question_camara, chamber_name=camara, k=K)
    print_chunks(chunks_camara, label="GraphRAG filtrado por cámara")
    resultados["patron_a_camara"] = {
        "question": question_camara,
        "chamber_filter": camara,
        "retrieved_ids": [c.intervention_id for c in chunks_camara],
    }

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN B — Agregación / contraste
# ─────────────────────────────────────────────────────────────────────────────

def demo_patron_b(
    graph: GraphRetriever,
    graph_client: MDBClient,
    pipeline_graph: RAGPipeline,
) -> dict:
    print_section("PATRÓN B — Agregación y contraste entre partidos/cámaras")
    resultados = {}

    # B.1 — Contraste por partido
    question_contraste = "¿Cuál es la postura sobre la migración e inmigración en Chile?"
    print(f"\n  B.1 Contraste por PARTIDO")
    print(f"  Pregunta: {question_contraste}")

    grouped_party = graph.retrieve_contrast_party(
        question_contraste,
        k_per_party=2,
        top_parties=5,
    )
    print_contrast(grouped_party, label="Top chunks por partido")

    # Construir contexto estructurado para el LLM
    context_contraste = graph.contrast_as_context(
        question_contraste, mode="party", k_per_party=2, top_parties=5
    )
    # Usar el pipeline directamente con contexto ya construido
    from openai import OpenAI
    from config.settings import (
        OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
        SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    )
    llm = OpenAI(api_key=OPENAI_API_KEY)
    user_msg = USER_PROMPT_TEMPLATE.format(
        context=context_contraste,
        question=f"Compara cómo distintos partidos políticos abordan el tema de la migración. "
                 f"Señala diferencias y coincidencias.",
    )
    completion = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    answer_contraste = completion.choices[0].message.content or ""
    print(f"\n  RESPUESTA LLM (contraste):\n  {answer_contraste[:600]}…")

    resultados["patron_b_contraste_partido"] = {
        "question": question_contraste,
        "parties_found": list(grouped_party.keys()),
        "answer": answer_contraste,
    }

    # B.2 — Contraste por cámara
    question_camara = "¿Cómo se ha debatido el proyecto de ley de salud mental?"
    print(f"\n  B.2 Contraste por CÁMARA")
    print(f"  Pregunta: {question_camara}")

    grouped_chamber = graph.retrieve_contrast_chamber(question_camara, k_per_chamber=2)
    print_contrast(grouped_chamber, label="Chunks por cámara")
    resultados["patron_b_contraste_camara"] = {
        "question": question_camara,
        "chambers_found": list(grouped_chamber.keys()),
    }

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN C — Atributo temporal / numérico
# ─────────────────────────────────────────────────────────────────────────────

def demo_patron_c(graph: GraphRetriever, pipeline_graph: RAGPipeline) -> dict:
    print_section("PATRÓN C — Atributo temporal y cohorte etaria")
    resultados = {}

    # C.1 — Cohorte etaria: ¿difieren los jóvenes de los mayores en el discurso?
    question_edad = "¿Qué se dice sobre el cambio climático y medio ambiente?"
    print(f"\n  C.1 Por COHORTE ETARIA")
    print(f"  Pregunta: {question_edad}")
    print(f"  Parlamentarios jóvenes (nacidos 1980–2000):")

    chunks_jovenes = graph.retrieve_by_age_cohort(
        question_edad, birth_year_min=1980, birth_year_max=2000, k=K
    )
    print_chunks(chunks_jovenes, label="Generación joven")

    print(f"  Parlamentarios mayores (nacidos 1945–1965):")
    chunks_mayores = graph.retrieve_by_age_cohort(
        question_edad, birth_year_min=1945, birth_year_max=1965, k=K
    )
    print_chunks(chunks_mayores, label="Generación mayor")

    resultados["patron_c_edad"] = {
        "question": question_edad,
        "jovenes_ids": [c.intervention_id for c in chunks_jovenes],
        "mayores_ids": [c.intervention_id for c in chunks_mayores],
    }

    # C.2 — Rango temporal: antes y después de un evento
    question_temporal = "¿Qué se ha discutido sobre la violencia en La Araucanía?"
    print(f"\n  C.2 Por RANGO DE FECHAS")
    print(f"  Pregunta: {question_temporal}")

    # Período 1: intervenciones antiguas
    chunks_antes = graph.retrieve_by_date_range(
        question_temporal,
        date_start="2018-01-01",
        date_end="2019-12-31",
        k=K,
    )
    print_chunks(chunks_antes, label="2018–2019")

    # Período 2: intervenciones recientes
    chunks_despues = graph.retrieve_by_date_range(
        question_temporal,
        date_start="2022-01-01",
        date_end="2023-12-31",
        k=K,
    )
    print_chunks(chunks_despues, label="2022–2023")

    resultados["patron_c_temporal"] = {
        "question": question_temporal,
        "ids_2018_2019": [c.intervention_id for c in chunks_antes],
        "ids_2022_2023": [c.intervention_id for c in chunks_despues],
    }

    # C.3 — Evolución temporal agrupada
    print(f"\n  C.3 EVOLUCIÓN TEMPORAL del discurso")
    evol = graph.retrieve_temporal_evolution(question_temporal, k_per_year=2)
    for year, chunks in evol.items():
        print(f"    {year}: {len(chunks)} chunk(s) | scores: {[round(c.score,3) for c in chunks]}")

    resultados["patron_c_evolucion"] = {
        "question": question_temporal,
        "years": {y: [c.intervention_id for c in chs] for y, chs in evol.items()},
    }

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 2.2 — Comparación GraphRAG vs RAG Denso
# ─────────────────────────────────────────────────────────────────────────────

def demo_comparacion(
    dense: DenseRetriever,
    graph: GraphRetriever,
    pipeline_dense: RAGPipeline,
    pipeline_graph: RAGPipeline,
) -> dict:
    """
    Muestra un caso donde GraphRAG es claramente mejor que el RAG denso.

    Caso elegido:
        "¿Qué opina la UDI sobre el impuesto a las grandes fortunas?"

    Por qué el denso falla aquí:
        El embedding de esta pregunta captura el tema "impuesto grandes fortunas"
        correctamente, pero no puede filtrar por partido. El denso devuelve
        intervenciones de CUALQUIER partido que hablen del tema, mezclando
        posturas favorables (izquierda) con críticas (derecha).
        Esto puede hacer que la respuesta del LLM mezcle posturas o atribuya
        argumentos equivocados a la UDI.

    Por qué GraphRAG gana:
        El filtro WHERE pp.name = "UDI" garantiza que el contexto contiene
        SOLO intervenciones de ese partido. El LLM recibe señal limpia y
        puede caracterizar fielmente la postura de la UDI.
    """
    print_section("PARTE 2.2 — Comparación: Dense RAG vs GraphRAG")

    question = "¿Qué opina la UDI sobre el impuesto a las grandes fortunas?"
    partido   = "Partido Unión Demócrata Independiente"   # Unión Demócrata Independiente

    print(f"\n  Pregunta: {question}")
    print(f"  Partido bajo análisis: {partido}")
    print(
        "\n  HIPÓTESIS: el RAG denso puede traer intervenciones de partidos\n"
        "  que apoyan el impuesto (izquierda), contaminando el contexto\n"
        "  sobre la postura de la UDI. GraphRAG filtra por partido en MDB."
    )

    # ── RAG Denso ────────────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  [DENSO] Recuperando top-{K} sin filtro de partido…")
    resp_dense = pipeline_dense.answer(question, k=K)
    print(f"\n  Chunks recuperados por el DENSO:")
    for c in resp_dense.retrieved_chunks:
        print(f"    [INT-{c.intervention_id}] partido={c.party} | score={c.score:.4f}")
    print(f"\n  RESPUESTA DENSA:\n  {resp_dense.answer[:600]}")

    # ── GraphRAG ─────────────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  [GRAPH] Recuperando top-{K} con filtro partido='{partido}'…")
    chunks_graph = graph.retrieve_by_party(question, party_name=partido, k=K)
    print(f"\n  Chunks recuperados por GraphRAG:")
    for c in chunks_graph:
        print(f"    [INT-{c.intervention_id}] partido={c.party} | score={c.score:.4f}")

    # Generar respuesta con los chunks filtrados
    from openai import OpenAI
    from config.settings import (
        OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
        LLM_TEMPERATURE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    )
    context_graph = "\n\n---\n\n".join(c.to_context_str() for c in chunks_graph)
    llm = OpenAI(api_key=OPENAI_API_KEY)
    completion = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    context=context_graph, question=question
                ),
            },
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    answer_graph = completion.choices[0].message.content or ""
    print(f"\n  RESPUESTA GRAPHRAG:\n  {answer_graph[:600]}")

    # ── Análisis del porqué ───────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  ANÁLISIS COMPARATIVO:")

    partidos_denso = list({c.party for c in resp_dense.retrieved_chunks if c.party})
    partidos_graph = list({c.party for c in chunks_graph if c.party})

    print(f"  Partidos en contexto DENSO  : {partidos_denso}")
    print(f"  Partidos en contexto GRAPH  : {partidos_graph}")

    if len(partidos_denso) > 1:
        print(
            f"\n  ⚠ El RAG denso mezcló {len(partidos_denso)} partidos distintos.\n"
            f"  El LLM no puede atribuir la postura correctamente a la UDI.\n"
            f"  GraphRAG entrega contexto puro de la UDI → respuesta más precisa."
        )
    else:
        print(
            f"\n  En este caso ambas estrategias coincidieron en el partido,\n"
            f"  pero GraphRAG provee la garantía estructural de que siempre filtra.\n"
            f"  Considera probar con partidos de menor presencia en los datos."
        )

    return {
        "pregunta": question,
        "partido_filtro": partido,
        "denso": {
            "partidos_en_contexto": partidos_denso,
            "answer": resp_dense.answer,
            "ids": [c.intervention_id for c in resp_dense.retrieved_chunks],
        },
        "graph": {
            "partidos_en_contexto": partidos_graph,
            "answer": answer_graph,
            "ids": [c.intervention_id for c in chunks_graph],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("PARTE 2 — GraphRAG sobre MillenniumDB")
    print(f"  Modelo LLM  : gpt-4o-mini")
    print(f"  Top-K       : {K}")
    print(f"  Recuperador : GraphRetriever (grafo + HNSW)")
    print("=" * 70)

    log = {
        "parte"    : 2,
        "estrategia": "GraphRAG",
        "top_k"    : K,
        "llm_model": "gpt-4o-mini",
        "timestamp": datetime.now().isoformat(),
        "resultados": {},
    }

    with MDBClient() as client:
        dense        = DenseRetriever(client=client, top_k=K, use_metadata=True)
        graph        = GraphRetriever(client=client, top_k=K)
        pipeline_dense = RAGPipeline(dense, retriever_name="dense", top_k=K)
        pipeline_graph = RAGPipeline(graph, retriever_name="graph", top_k=K)

        # Patrón A
        try:
            log["resultados"]["patron_a"] = demo_patron_a(graph, pipeline_graph)
        except Exception as e:
            print(f"[ERROR] Patrón A: {e}")
            log["resultados"]["patron_a"] = {"error": str(e)}

        # Patrón B
        try:
            log["resultados"]["patron_b"] = demo_patron_b(graph, client, pipeline_graph)
        except Exception as e:
            print(f"[ERROR] Patrón B: {e}")
            log["resultados"]["patron_b"] = {"error": str(e)}

        # Patrón C
        try:
            log["resultados"]["patron_c"] = demo_patron_c(graph, pipeline_graph)
        except Exception as e:
            print(f"[ERROR] Patrón C: {e}")
            log["resultados"]["patron_c"] = {"error": str(e)}

        # Comparación (Parte 2.2)
        try:
            log["resultados"]["comparacion_2_2"] = demo_comparacion(
                dense, graph, pipeline_dense, pipeline_graph
            )
        except Exception as e:
            print(f"[ERROR] Comparación 2.2: {e}")
            log["resultados"]["comparacion_2_2"] = {"error": str(e)}

    # Guardar log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Log guardado en: {LOG_FILE}")
    print("\n" + "=" * 70)
    print("Parte 2 completada.")
    print("=" * 70)


if __name__ == "__main__":
    main()

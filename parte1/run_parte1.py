"""
parte1/run_parte1.py  —  Parte 1.3
------------------------------------
Script principal de la Parte 1: RAG Denso.

Ejecutar:
    cd rag_legislativo
    python parte1/run_parte1.py

Qué hace:
  1. Conecta a MillenniumDB.
  2. Inicializa el recuperador denso y el pipeline RAG.
  3. Ejecuta un conjunto de preguntas de prueba.
  4. Imprime resultados con contexto y respuesta.
  5. Guarda un log JSON con todas las respuestas.

Preguntas de prueba (Parte 1.3):
  Se eligieron preguntas que cubren distintos tipos:
  - Temáticas transversales (pensiones, salud)
  - Proyectos de ley específicos
  - Posturas generales de la cámara
  Nota: el RAG denso responde bien preguntas temáticas pero puede fallar
  en preguntas que requieren filtrar por partido o fecha → lo veremos en Parte 2.
"""

from __future__ import annotations
import sys, os, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdb_client import MDBClient
from parte1.dense_retriever import DenseRetriever
from parte1.rag_pipeline import RAGPipeline
from config.settings import TOP_K

# ── Preguntas de prueba ───────────────────────────────────────────────────────
# Elegidas para mostrar fortalezas y limitaciones del RAG denso.

TEST_QUESTIONS = [
    # 1. Temática amplia — el denso debería recuperar bien
    "¿Cuáles son los principales argumentos que se han dado en el congreso "
    "sobre la reforma al sistema de pensiones en Chile?",

    # 2. Temática de salud pública
    "¿Qué han dicho los parlamentarios sobre el acceso a medicamentos "
    "y el rol de la industria farmacéutica?",

    # 3. Medio ambiente y recursos naturales
    "¿Cómo se ha debatido en el congreso el tema del agua y la crisis hídrica?",

    # 4. Pregunta con actor específico — el denso puede fallar si no hay
    #    suficiente señal semántica sobre el partido
    "¿Qué postura tiene la UDI respecto al aumento del salario mínimo?",

    # 5. Pregunta temporal — el denso ignora el tiempo
    "¿Qué intervenciones se han dado sobre seguridad ciudadana "
    "y control de armas en el último período legislativo?",

    # 6. Contraste de posturas — interesante para Parte 4
    "¿Cuáles son las diferencias entre las posturas de derecha e izquierda "
    "respecto al rol del Estado en la economía?",
]


# ── Configuración del experimento ─────────────────────────────────────────────
K = TOP_K          # k=5, fijo en todas las partes
LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"resultados_parte1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)


def main() -> None:
    print("=" * 70)
    print("PARTE 1 — RAG Denso sobre MillenniumDB")
    print(f"  Modelo LLM   : gpt-4o-mini")
    print(f"  Top-K        : {K}")
    print(f"  Recuperador  : DenseRetriever (HNSW en MDB)")
    print("=" * 70)

    with MDBClient() as client:
        # Inicializar componentes
        retriever = DenseRetriever(client=client, top_k=K, use_metadata=True)
        pipeline  = RAGPipeline(retriever=retriever, retriever_name="dense", top_k=K)

        resultados = []

        for idx, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n{'─'*70}")
            print(f"PREGUNTA {idx}/{len(TEST_QUESTIONS)}")
            print(f"{'─'*70}")

            try:
                resp = pipeline.answer(question, k=K)
                resp.print_summary()

                # Guardar para log
                resultados.append({
                    "id"               : idx,
                    "question"         : resp.question,
                    "answer"           : resp.answer,
                    "retriever"        : resp.retriever_name,
                    "top_k"            : resp.top_k,
                    "prompt_tokens"    : resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "retrieved_ids"    : [
                        c.intervention_id for c in resp.retrieved_chunks
                    ],
                    "scores"           : [
                        round(c.score, 4) for c in resp.retrieved_chunks
                    ],
                })

            except Exception as exc:
                print(f"  [ERROR] Pregunta {idx}: {exc}")
                resultados.append({
                    "id"      : idx,
                    "question": question,
                    "error"   : str(exc),
                })

    # Guardar log JSON
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "parte"       : 1,
                "estrategia"  : "RAG denso",
                "top_k"       : K,
                "llm_model"   : "gpt-4o-mini",
                "timestamp"   : datetime.now().isoformat(),
                "resultados"  : resultados,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[INFO] Log guardado en: {LOG_FILE}")
    print("\n" + "=" * 70)
    print("Parte 1 completada.")
    print("=" * 70)

    # ── Análisis rápido ──────────────────────────────────────────────────────
    print("\n[ANÁLISIS RÁPIDO]")
    print(f"  Total preguntas  : {len(resultados)}")
    exitosas = [r for r in resultados if "error" not in r]
    print(f"  Respuestas OK    : {len(exitosas)}")
    if exitosas:
        avg_ptk = sum(r["prompt_tokens"]     for r in exitosas) / len(exitosas)
        avg_ctk = sum(r["completion_tokens"] for r in exitosas) / len(exitosas)
        print(f"  Tokens promedio  : prompt={avg_ptk:.0f} | completion={avg_ctk:.0f}")

    print(
        "\n[NOTA] Las preguntas 4 y 5 suelen ser las más débiles con RAG denso:\n"
        "  - Pregunta 4: requiere filtrar por partido → GraphRAG (Parte 2)\n"
        "  - Pregunta 5: requiere filtro temporal → GraphRAG con fecha (Parte 2)"
    )


if __name__ == "__main__":
    main()

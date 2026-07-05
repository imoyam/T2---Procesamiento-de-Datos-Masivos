import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdb_client import MDBClient
from parte1.dense_retriever import DenseRetriever
from parte1.rag_pipeline import RAGPipeline


PREGUNTAS = [
    "¿Cuáles son los principales argumentos sobre la reforma al sistema de pensiones en Chile?",
    "¿Qué han dicho los parlamentarios sobre el acceso a medicamentos?",
    "¿Qué postura tiene la UDI respecto al aumento del salario mínimo?",
    "¿Cuáles son las diferencias entre la derecha e izquierda respecto al rol del Estado?"
]


def main() -> None:
    print("=" * 70)
    print("Iniciando Parte 1: RAG Denso...")
    print("=" * 70)

    resultados = []

    with MDBClient() as cliente:
        retriever = DenseRetriever(client=cliente)
        pipeline = RAGPipeline(retriever=retriever)

        for idx, pregunta in enumerate(PREGUNTAS, 1):
            print(f"\nProcesando {idx}/{len(PREGUNTAS)}...")
            print(f"Pregunta: {pregunta}")

            try:
                resp = pipeline.answer(pregunta)

                #print("\nRespuesta:")
                #print(resp.answer)

                #print("\nChunks recuperados:")
                for i, chunk in enumerate(resp.retrieved_chunks, 1):
                    print(f"{i}. INT-{chunk.intervention_id} | partido: {chunk.party} | score: {chunk.score}")
                    print(chunk.text[:300])
                    print("-" * 50)

                resultados.append({
                    "pregunta": resp.question,
                    "respuesta": resp.answer,
                    "chunks": [
                        {
                            "intervention_id": c.intervention_id,
                            "chunk_id": c.chunk_id,
                            "party": c.party,
                            "score": c.score,
                            "text": c.text,
                        }
                        for c in resp.retrieved_chunks
                    ]
                })

            except Exception as e:
                print(f"[ERROR] {e}")

    with open("resultados_parte1.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Resultados guardados en resultados_parte1.json")


if __name__ == "__main__":
    main()
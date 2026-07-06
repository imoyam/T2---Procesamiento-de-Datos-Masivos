import argparse
import csv
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
from parte2.graph_retriever import GraphRetriever
from part3.sparse_retriever import SparseRetriever
from part3.hybrid_retriever import HybridRetriever
from utils.mdb_client import MDBClient

PREGUNTAS = [
    "1. ¿Qué argumentos a favor y en contra surgieron en el debate sobre la reducción de la jornada laboral a 40 horas semanales, considerando aspectos como la conciliación entre la vida familiar y el trabajo, la productividad y el efecto sobre las pymes?",
    "2. En torno a la despenalización del aborto, la interrupción voluntaria del embarazo, los derechos reproductivos y la objeción de conciencia, ¿cómo se diferencia la manera en que se han pronunciado los legisladores hombres frente a las legisladoras mujeres?",
    "3. Sobre el impacto ambiental de la minería, el royalty minero, la escasez hídrica y el desarrollo de las comunidades del norte, ¿qué diferencias se observan entre lo que plantean quienes representan distritos con comunas mineras y quienes representan zonas sin actividad minera?",
    "4. En materia de migración irregular, expulsión de extranjeros, crisis migratoria en la frontera norte y seguridad fronteriza, ¿qué posturas defienden los distintos partidos políticos y en qué se contraponen entre sí?",
    "5. Identifica las intervenciones en que se mencionan de forma explícita el “CAE” (Crédito con Aval del Estado), la “condonación” de la deuda educativa y el nuevo “FES” (Financiamiento para la Educación Superior); ¿en qué contextos aparecen exactamente estos términos?",
    "6. ¿Cómo se ha discutido el uso de la fuerza policial y la legítima defensa de Carabineros, en particular en aquellas intervenciones que se refieren específicamente a la Ley Naín-Retamal?",
    "7. ¿Qué proyectos de ley sobre la muerte digna, la eutanasia y los cuidados paliativos terminaron archivados, y qué se argumentó en las intervenciones asociadas a esos proyectos?",
    "8. ¿Cuál es la postura específica del Partido Comunista de Chile frente al despliegue de las Fuerzas Armadas en infraestructura crítica y las Reglas de Uso de la Fuerza (RUF) en zonas urbanas?",
    "9. Identifica las intervenciones que debaten explícitamente la 'Ley Karin' y la ratificación del 'Convenio 190' de la OIT. ¿Qué exigencias preventivas específicas se mencionan para los empleadores?"
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Parte 4: Generación de RAG y exportación para evaluación manual.")
    parser.add_argument("--k", type=int, default=5, help="Top-k para recuperar y evaluar (Recomendado: 5 para no evaluar demasiados textos a mano).")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = base_path / f"respuestas_llm_parte4_{timestamp}.json"
    csv_path = base_path / f"evaluacion_manual_{timestamp}.csv"

    print("=" * 80)
    print("PARTE 4 - Preparación de Evaluación Manual y Generación (Sin LLM-as-a-judge)")
    print(f"Top-K: {args.k}")
    print("=" * 80)

    log_respuestas = []
    filas_csv = [["id_pregunta", "pregunta", "estrategia", "rank", "intervention_id", "texto", "nota_manual_1_a_7"]]

    with MDBClient() as client:
        dense = DenseRetriever(client=client, top_k=args.k)
        graph = GraphRetriever(client=client, top_k=args.k)
        sparse = SparseRetriever(client=client, top_k=args.k, auto_build=False)
        hybrid = HybridRetriever(client=client, dense=dense, sparse=sparse, top_k=args.k)

        pipelines = {
            "1_Dense": RAGPipeline(dense, "Dense", args.k),
            "2_GraphRAG": RAGPipeline(graph, "GraphRAG", args.k),
            "3_Sparse": RAGPipeline(sparse, "Sparse", args.k),
            "4_Hybrid": RAGPipeline(hybrid, "Hybrid", args.k),
        }

        for q_idx, question in enumerate(PREGUNTAS, start=1):
            print(f"\n[{q_idx}/{len(PREGUNTAS)}] Procesando: {question[:80]}...")
            
            respuestas_estrategias = {}

            for nombre_est, pipeline in pipelines.items():
                print(f"  -> Recuperando y generando con: {nombre_est}")
                try:
                    if nombre_est == "2_GraphRAG" and q_idx == 4:
                        chunks = graph.retrieve_contrast_party(question, k_per_party=1)
                        #aplanamos el diccionario
                        chunks = [c for sublist in chunks.values() for c in sublist][:args.k]
                        context = graph.contrast_as_context(question, mode="party", k_per_party=1)
                        #llamada la LLM 
                        user_msg = pipeline.user_prompt_template.format(context=context, question=question) if hasattr(pipeline, 'user_prompt_template') else f"Contexto:\n{context}\n\nPregunta:\n{question}"
                        import openai
                        client_oai = openai.OpenAI()
                        comp = client_oai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": user_msg}])
                        ans = comp.choices[0].message.content
                        resp = type('obj', (object,), {'question': question, 'answer': ans, 'retrieved_chunks': chunks})
                    else:
                        resp = pipeline.answer(question, k=args.k)

                    respuestas_estrategias[nombre_est] = resp.answer

                    for rank, chunk in enumerate(resp.retrieved_chunks, start=1):
                        filas_csv.append([
                            f"Q{q_idx}", 
                            question, 
                            nombre_est, 
                            str(rank), 
                            chunk.intervention_id, 
                            chunk.text[:500].replace("\n", " ").replace(";", ","),
                            "" 
                        ])
                except Exception as e:
                    print(f"     [ERROR] {e}")

            log_respuestas.append({
                "id_pregunta": f"Q{q_idx}",
                "pregunta": question,
                "respuestas_llm": respuestas_estrategias
            })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_respuestas, f, ensure_ascii=False, indent=2)
    #sacamos csv para llenarlo manualmente
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(filas_csv)

    print("\n" + "=" * 80)
    print("¡Generación exitosa!")
    print(f"1. Abre el archivo: {csv_path}")
    print("2. Lee los textos y llena la última columna ('nota_manual_1_a_7') con notas del 1 al 7.")
    print(f"3. Las respuestas finales del LLM están en: {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
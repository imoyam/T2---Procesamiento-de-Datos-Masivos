import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdb_client import MDBClient
from parte1.dense_retriever import RecuperadorDenso
from parte1.rag_pipeline import PipelineRAG

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

    with MDBClient() as cliente:
        recuperador = RecuperadorDenso(cliente=cliente)
        pipeline = PipelineRAG(recuperador=recuperador)
        resultados = []
        for idx, q in enumerate(PREGUNTAS, 1):
            print(f"\nProcesando {idx}/{len(PREGUNTAS)}...")
            try:
                resp = pipeline.responder(q)
                resp.imprimir()
                resultados.append({
                    "pregunta": resp.pregunta, 
                    "respuesta": resp.respuesta,
                    "tokens": f"{resp.tokens_prompt} prompt / {resp.tokens_respuesta} completion"
                })
            except Exception as e:
                print(f" [ERROR] {e}")

    #exportación a JSON
    with open("resultados_parte1.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Resultados guardados en resultados_parte1.json")

if __name__ == "__main__":
    main()
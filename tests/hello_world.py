import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mdb_client import MDBClient
from utils.embedder import embed_query, vec_to_list

CONSULTAS = [
    ("Nodos Intervención", "MATCH (?i:Intervention) RETURN COUNT(?i)"),
    ("Nodos Persona", "MATCH (?p:Person) RETURN COUNT(?p)"),
    ("Nodos Posición", "MATCH (?pos:Position) RETURN COUNT(?pos)"),
    ("Nodos Partido", "MATCH (?pp:PoliticalParty) RETURN COUNT(?pp)"),
    ("Nodos Sesión", "MATCH (?s:Session) RETURN COUNT(?s)"),
    ("Nodos Proyecto Ley", "MATCH (?b:Bill) RETURN COUNT(?b)"),
    ("Relaciones Autoría", "MATCH (?i:Intervention)-[:IsDeliveredBy]->(?pos:Position) RETURN COUNT(?i)"),
]

def probar_hnsw(cliente: MDBClient) -> None:
    #prueba semantica
    query_texto = "pensiones en Chile"
    vector = vec_to_list(embed_query(query_texto))
    query = f"MATCH (i) RETURN i['id'], SIMILARITY(i['vector'], {vector}) LIMIT 5" #ARREGLAR ESTA QUERY XD
    try:
        resultados = cliente.run(query)
        print(f"\nResultado HNSW para '{query_texto}':")
        for r in resultados:
            print(f" -> {r}")
    except Exception as e:
        print(f"\nError en HNSW: {e}")

def main() -> None:
    print("Iniciando validación de MillenniumDB...")

    with MDBClient() as cliente:
        #loop de CONSULTAS de conteo
        for descripcion, consulta in CONSULTAS:
            print(f"\nVerificando: {descripcion}")
            try:
                filas = cliente.run(consulta)
                for fila in filas:
                    print(f" -> {fila}")
            except Exception as e:
                print(f" Error: {e}")

        #prueba del indice vectorial
        probar_hnsw(cliente)

    print("\nValidación finalizada.")

if __name__ == "__main__":
    main()
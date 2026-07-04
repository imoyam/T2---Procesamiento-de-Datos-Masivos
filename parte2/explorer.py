import sys, os, csv
from io import StringIO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mdb_client import MDBClient

def obtener_resumen(client, query):
    resultados = client.run(query)
    unicos = set()
    for fila in resultados:
        if not fila: continue
        # Extraemos los valores aplastados por el cliente
        raw_vals = list(fila.values())[0]
        try:
            parsed = next(csv.reader(StringIO(raw_vals)))
            # Limpiamos los corchetes feos de las etiquetas
            clean_vals = [v.replace('[', '').replace(']', '').replace('"', '') for v in parsed]
            unicos.add(" | ".join(clean_vals))
        except:
            pass
    return unicos

def main():
    print("=" * 60)
    print("RADIOGRAFÍA EXACTA DEL GRAFO")
    print("=" * 60)
    
    with MDBClient() as client:
        print("\n1. ¿Hacia dónde apunta una Intervención?")
        for x in obtener_resumen(client, "MATCH (?i:Intervention)-[?e]->(?n) RETURN DISTINCT TYPE(?e), LABELS(?n)"): 
            print(f"   Intervention -> [{x.split(' | ')[0]}] -> {x.split(' | ')[1]}")
            
        print("\n2. ¿Qué apunta hacia una Intervención?")
        for x in obtener_resumen(client, "MATCH (?n)-[?e]->(?i:Intervention) RETURN DISTINCT LABELS(?n), TYPE(?e)"): 
            print(f"   {x.split(' | ')[0]} -> [{x.split(' | ')[1]}] -> Intervention")

        print("\n3. ¿Cuáles son TODAS las relaciones del sistema?")
        for x in obtener_resumen(client, "MATCH ()-[?e]->() RETURN DISTINCT TYPE(?e)"): 
            print(f"   - {x}")

if __name__ == "__main__":
    main()
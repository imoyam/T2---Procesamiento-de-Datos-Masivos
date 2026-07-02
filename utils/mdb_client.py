import requests
from typing import Any, Iterator
from config.settings import MDB_HOST, MDB_PORT

class MDBClient:
    def __init__(self, host: str = MDB_HOST, port: int = MDB_PORT) -> None:
        self.url = f"http://{host}:{port}"
        print(f"[MDB] Cliente HTTP listo en {self.url}")
    def __enter__(self) -> "MDBClient": return self
    def __exit__(self, *_) -> None: pass

    def run(self, query: str) -> list[dict[str, Any]]:
        try:
            respuesta = requests.post(self.url, data=query.encode('utf-8'), headers={"Accept": "application/json"})
            
            if respuesta.status_code != 200:
                raise ValueError(f"HTTP {respuesta.status_code}: {respuesta.text}")
            
            if not respuesta.text.strip():
                return []

            #parseo de formato TSV (retorno tabular)
            lineas = respuesta.text.strip().split('\n')
            columnas = [c.strip() for c in lineas[0].split('\t')]
            
            return [dict(zip(columnas, [v.strip() for v in l.split('\t')])) for l in lineas[1:]]

        except Exception as e:
            print(f"[MDB] Error en query:\n{query}\n→ {e}")
            raise

    def run_iter(self, query: str) -> Iterator[dict]:
        for fila in self.run(query):
            yield fila